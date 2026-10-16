"""Benchmark GPU multipiattaforma via WebGPU (wgpu: Vulkan, Metal o DirectX 12).

Misura due grandezze che determinano le prestazioni di un LLM:
- banda di memoria effettiva (GB/s): limita la velocità di generazione dei token;
- potenza di calcolo FP32 (TFLOPS): limita la lettura del prompt e dei documenti.
"""

import time

COPY_SHADER = """
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * 256u;
    let n = arrayLength(&src);
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        dst[i] = src[i];
        i = i + stride;
    }
}
"""

FMA_ITERS = 512
FMA_CHAINS = 8
FMA_SHADER = """
@group(0) @binding(0) var<storage, read_write> data: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = data[gid.x & 1023u];
    let m = data[1024u];
    let c = data[1025u];
    var x0 = s; var x1 = s + 1.0; var x2 = s + 2.0; var x3 = s + 3.0;
    var x4 = s + 4.0; var x5 = s + 5.0; var x6 = s + 6.0; var x7 = s + 7.0;
    for (var i = 0u; i < ITERSu; i = i + 1u) {
        x0 = fma(x0, m, c); x1 = fma(x1, m, c); x2 = fma(x2, m, c); x3 = fma(x3, m, c);
        x4 = fma(x4, m, c); x5 = fma(x5, m, c); x6 = fma(x6, m, c); x7 = fma(x7, m, c);
    }
    let r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    if (dot(r, vec4<f32>(1.0)) == -12345.0) {  // mai vero: impedisce al compilatore di eliminare il ciclo
        data[gid.x & 1023u] = r;
    }
}
""".replace("ITERS", str(FMA_ITERS))

MB = 1024 ** 2


def _make_pipeline(device, code):
    module = device.create_shader_module(code=code)
    return device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})


def _timed(device, pipeline, bind_group, groups, repeats, sync_buf):
    enc = device.create_command_encoder()
    for _ in range(repeats):
        p = enc.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(groups)
        p.end()
    cmd = enc.finish()
    t0 = time.perf_counter()
    device.queue.submit([cmd])
    device.queue.read_buffer(sync_buf, 0, 4)  # attende la fine del lavoro sulla GPU
    return time.perf_counter() - t0


def _per_dispatch(device, pipeline, bind_group, groups, sync_buf, extra):
    """Tempo per singola esecuzione, togliendo il costo fisso di invio e sincronizzazione."""
    base = min(_timed(device, pipeline, bind_group, groups, 1, sync_buf) for _ in range(3))
    many = min(_timed(device, pipeline, bind_group, groups, 1 + extra, sync_buf) for _ in range(3))
    return max((many - base) / extra, 1e-9)


def run(vram_hint_bytes=None, log=print):
    """Esegue il benchmark sulla GPU più potente. Restituisce un dict oppure None."""
    try:
        import numpy as np
        import wgpu
    except ImportError:
        log("  wgpu/numpy non disponibili: salto il benchmark GPU.")
        return None
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as e:  # noqa: BLE001 - qualunque errore del driver
        log(f"  Nessun adattatore WebGPU utilizzabile ({e}).")
        return None
    if adapter is None:
        return None
    info = adapter.info
    result = {
        "adapter": info.get("device") or info.get("description") or "?",
        "backend": info.get("backend_type"),
        "adapter_type": info.get("adapter_type"),
        "bandwidth_gbs": None,
        "fp32_tflops": None,
        "shader_f16": "shader-f16" in adapter.features,
    }
    if result["adapter_type"] == "CPU":
        log("  L'unico adattatore disponibile è software (CPU): niente benchmark GPU.")
        result["skipped"] = "software"
        return result

    limit = adapter.limits["max-storage-buffer-binding-size"]
    size = min(256 * MB, limit, adapter.limits["max-buffer-size"])
    if vram_hint_bytes:
        size = min(size, max(16 * MB, vram_hint_bytes // 8))
    size -= size % 16
    device = adapter.request_device_sync(required_limits={
        "max-storage-buffer-binding-size": size, "max-buffer-size": size})
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    try:
        # --- banda di memoria
        log(f"  Banda memoria su {result['adapter']} ({result['backend']}), buffer {size // MB} MB...")
        src = device.create_buffer_with_data(data=np.ones(size // 4, dtype=np.float32), usage=usage)
        dst = device.create_buffer(size=size, usage=usage)
        copy_pipe = _make_pipeline(device, COPY_SHADER)
        copy_bg = device.create_bind_group(layout=copy_pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": src, "offset": 0, "size": size}},
            {"binding": 1, "resource": {"buffer": dst, "offset": 0, "size": size}},
        ])
        groups = 4096
        _timed(device, copy_pipe, copy_bg, groups, 3, dst)  # riscaldamento
        t = _per_dispatch(device, copy_pipe, copy_bg, groups, dst, extra=20)
        result["bandwidth_gbs"] = round(2 * size / t / 1e9, 1)
        src.destroy()
        dst.destroy()

        # --- calcolo FP32
        log("  Potenza di calcolo FP32...")
        seed = np.full((2048, 4), 0.5, dtype=np.float32)
        seed[1024] = 0.9999
        seed[1025] = 0.0001
        buf = device.create_buffer_with_data(data=seed, usage=usage)
        fma_pipe = _make_pipeline(device, FMA_SHADER)
        fma_bg = device.create_bind_group(layout=fma_pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": seed.nbytes}}])
        # Si parte piccoli e si cresce: un'esecuzione lunga su una GPU debole
        # farebbe scattare il watchdog del driver (TDR su Windows, ~2 s).
        groups = 64
        while groups < 65535:
            if _timed(device, fma_pipe, fma_bg, groups, 1, buf) > 0.08:
                break
            groups = min(groups * 4, 65535)
        t = _per_dispatch(device, fma_pipe, fma_bg, groups, buf, extra=5)
        flops = groups * 256 * FMA_ITERS * FMA_CHAINS * 4 * 2
        result["fp32_tflops"] = round(flops / t / 1e12, 2)
        buf.destroy()
    except Exception as e:  # noqa: BLE001
        log(f"  Benchmark GPU interrotto: {e}")
        result["error"] = str(e)
    return result
