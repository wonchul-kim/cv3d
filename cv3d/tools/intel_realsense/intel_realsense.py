import os
import os.path as osp
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
import yaml 
import json
import numpy as np
import time
from datetime import datetime

try:
    import pyrealsense2 as rs
except Exception as e:
    print(f"[ERROR] While importing `pyrealsense2`, {repr(e)}")
    print(f"   Going to install `pyrealsense2`...")
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "pyrealsense2"])
    import pyrealsense2 as rs

try:
    import cv2
except Exception as e:
    print(f"[ERROR] While importing `imageio`, {repr(e)}")
    print(f"   Going to install `pyrealsense2`...")
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python"])
    import cv2

    

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def profile_to_dict(p: rs.stream_profile):
            d = {
                "stream": str(p.stream_type()),
                "fmt": str(p.format()),
                "index": p.stream_index(),
                "fps": p.fps()
            }
            # 비디오 스트림이면 해상도/내부파라미터 추출
            try:
                v = p.as_video_stream_profile()
                i = v.get_intrinsics()
                d.update({
                    "width": v.width(),
                    "height": v.height(),
                    "intrinsics": {
                        "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                        "coeffs": list(i.coeffs), "model": str(i.model)
                    }
                })
            except Exception:
                pass
            return d

class IntelRealSense:
    def __init__(self, config_file=ROOT / 'config.yaml'):
        
        if isinstance(config_file, str):
            assert osp.exists(config_file), RuntimeError(f"There is no required config-file at {config_file}")
        elif isinstance(config_file, Path):
            assert config_file.exists(), RuntimeError(f"There is no required config-file at {config_file}")
            config_file = str(config_file)
            
        with open(config_file, 'r') as yf:
            self._config = yaml.load(yf)
            
        os.makedirs(self._config['output_dir'], exist_ok=True)
        print(f"Created Output Directory at: {self._config['output_dir']}")
        
        self._set()
        
    def _set_config(self):
        if self._config['save_ply']:
            if not self._config['pointcloud']:
                print(f"[WARN] To save ply, `pointcloud`({self._config['pointcloud']}) must be true")
                print(f"    Now it is on")
                self._config['pointcloud'] = True
                
            try:
                import open3d as o3d
            except Exception as e:
                print(f"[ERROR] While importing `open3d`, {repr(e)}")
                print(f"   Going to install `open3d`...")
                import sys, subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "open3d"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.25"])
                import open3d as o3d
                    
            if not self._config['color']:
                print(f"[WARN] To save ply, `color`({self._config['color']}) must be true")
                print(f"    Now it is on")
                self._config['color'] = True
                
        if self._config['pointcloud']:
            if not self._config['depth']:
                print(f"[WARN] To get pointcloud, `depth`({self._config['depth']}) must be true")
                print(f"    Now it is on")
                self._config['depth'] = True
            
        if self._config['depth']:
            if not (self._config['infra1'] and self._config['infra2']):
                print(f"[WARN] To get depth, `infra1`({self._config['infra1']}) and `infra2`({self._config['infra2']}) must be true")
                print(f"    Now they are on")
                self._config['infra1'], self._config['infra2'] = True, True
            
        if self._config['color'] and self._config['depth']:
            print(f"[INFO] Since `color` and `depth` are true, `align_to_color` is also true")
            self._config['align_to_color'] = True
            
    def _set(self):
        
        self._rs_ctx = rs.context()
        self._rs_ctx_devices = self._rs_ctx.query_devices()
        assert len(self._rs_ctx_devices) != 0, RuntimeError(f"There is no such device detected for IntelRealSense")
        
        self._set_config()
        self._set_pipeline()
        
    def _set_pipeline(self):
        
        def find_stereo_sensor(dev: rs.device):
            for s in dev.sensors:
                # 이름 기준(주로 "Stereo Module") 또는 depth_sensor로 캐스팅 가능 여부
                name = s.get_info(rs.camera_info.name) if s.supports(rs.camera_info.name) else ""
                try:
                    _ = rs.depth_sensor(s)  # 캐스팅 성공하면 스테레오 모듈
                    return s, name
                except Exception:
                    pass
            return None, None
        
        self._rs_pipeline = rs.pipeline()
        self._rs_config = rs.config()
        if self._config['serial']:
            self._rs_config.enable_device(self._config['serial'])
            
        if self._config['bag']:
            self._rs_config.enable_record_to_file(self._config['bag'])
            
        ### stream setting -------------------------------------------------------------------------------
        if self._config['color']:
            # 컬러 포맷은 장치/해상도에 따라 RGB8/YUYV/MJPEG 제공 가능함
            self._rs_config.enable_stream(rs.stream.color, self._config['width'], self._config['height'], rs.format.bgr8, self._config['fps'])
            
        if self._config['depth']:
            self._rs_config.enable_stream(rs.stream.depth, self._config['width'], self._config['height'], rs.format.z16, self._config['fps'])

        if self._config['infra1']:
            self._rs_config.enable_stream(rs.stream.infrared, 1, self._config['width'], self._config['height'], rs.format.y8, self._config['fps'])

        if self._config['infra2']:
            self._rs_config.enable_stream(rs.stream.infrared, 2, self._config['width'], self._config['height'], rs.format.y8, self._config['fps'])


        ### pipeline
        pipe_profile = self._rs_pipeline.start(self._rs_config)
        dev = pipe_profile.get_device()
        self._serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "unknown"
        product_line = dev.get_info(rs.camera_info.product_line) if dev.supports(rs.camera_info.product_line) else "unknown"
        print(f"[INFO] Device serial={self._serial}, product_line={product_line}")

        stereo, name = find_stereo_sensor(dev)
        if stereo is None:
            raise RuntimeError("Stereo Module(IR) 센서를 찾지 못했습니다.")

        def opt_range(opt):
            r = stereo.get_option_range(opt)
            return r.min, r.max, r.step, r.default
        
        def clamp(v, lo, hi): return max(lo, min(hi, v))
        
        ### (1) 오토 노출 끄기 → 수동으로 올리기
        if stereo.supports(rs.option.enable_auto_exposure):
            stereo.set_option(rs.option.enable_auto_exposure, 0)
        
        ### (2) 노출(µs) & 게인 올리기
        if stereo.supports(rs.option.exposure):
            lo, hi, st, de = opt_range(rs.option.exposure)
            ### 30fps면 노출 상한이 짧습니다; 15/6fps로 낮추면 더 긴 노출 허용
            target_exposure = clamp(20000, lo, hi) # gain
            stereo.set_option(rs.option.exposure, target_exposure)
            print("exposure set:", target_exposure)

        if stereo.supports(rs.option.gain):
            lo, hi, st, de = opt_range(rs.option.gain)
            target_gain = clamp(50, lo, hi) # gain
            stereo.set_option(rs.option.gain, target_gain)
            print("gain set:", target_gain)

        # # (3) 에미터(레이저/패턴) 켜기 + 파워 올리기 → IR 대비/밝기 향상(실내에서 효과적)
        # if stereo.supports(rs.option.emitter_enabled):
        #     stereo.set_option(rs.option.emitter_enabled, 1)  # 1=ON
        # if stereo.supports(rs.option.laser_power):
        #     lo, hi, st, de = opt_range(rs.option.laser_power)
        #     stereo.set_option(rs.option.laser_power, clamp(hi, lo, hi))  # 상한으로

        # # (4) 프리셋 바꾸기(모드별 노출/필터 튠)
        # if stereo.supports(rs.option.visual_preset):
        #     # rs.rs400_visual_preset 에서 HIGH_ACCURACY/HIGH_DENSITY/HAND 같은 프리셋 사용 가능
        #     stereo.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)

        self._depth_scale = None
        for s in dev.sensors:
            try:
                ds = rs.depth_sensor(s)
                self._depth_scale = ds.get_depth_scale()
            except Exception:
                pass
            
        if self._depth_scale is None and self._config['depth']:
            print("[WARN] depth_scale을 가져오지 못했습니다. (장치/센서 확인)")

        # 정렬기
        self._rs_align = rs.align(rs.stream.color) if self._config['align_to_color'] else None
        
        # 포인트 클라우드 생성기
        self._rs_pointcloud_generator = rs.pointcloud() if (self._config['pointcloud'] or self._config['save_ply']) else None
        
        # 포인트 클라우드 생성기
        self._rs_colorizer = rs.colorizer() if self._config['depth'] else None

        # 프로파일/캘리브레이션 저장 (1회)
        # extrinsics: 각 스트림 -> color 기준
        calib = {"device": {"serial": self._serial, "product_line": product_line}, "streams": {}}
        for sp in pipe_profile.get_streams():
            try:
                vs = sp.as_video_stream_profile()
                k = f"{str(sp.stream_type())}_{sp.stream_index()}"
                calib["streams"][k] = {
                    "profile": profile_to_dict(sp),
                    "extrinsics_to_color": None
                }
            except Exception:
                pass
        
        try:
            color_sp = pipe_profile.get_stream(rs.stream.color)
            for sp in pipe_profile.get_streams():
                try:
                    ex = sp.get_extrinsics_to(color_sp)
                    k = f"{str(sp.stream_type())}_{sp.stream_index()}"
                    if k in calib["streams"]:
                        calib["streams"][k]["extrinsics_to_color"] = {
                            "rotation": list(ex.rotation),
                            "translation": list(ex.translation)
                        }
                except Exception:
                    pass
        except Exception:
            pass

        if self._depth_scale is not None:
            calib["depth_scale_m"] = self._depth_scale
            
        with open(osp.join(self._config['output_dir'], f"calibration_{self._serial}.json"), "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2, ensure_ascii=False)
            
        print(f"[INFO] Saved calibration info at {osp.join(self._config['output_dir'], f'calibration_{self._serial}.json')}")

        self._ddir = Path(osp.join(self._config['output_dir'], "depth"))
        self._cdir = Path(osp.join(self._config['output_dir'], "color"))
        self._pcdir = Path(osp.join(self._config['output_dir'], "pointcloud"))
        self._idir1 = Path(osp.join(self._config['output_dir'], "infra1"))
        self._idir2 = Path(osp.join(self._config['output_dir'], "infra2"))
        for p in [self._ddir, self._cdir, self._pcdir, self._idir1, self._idir2]:
            os.makedirs(p, exist_ok=True)

        
    def _get_device_info(self, device: rs.device):
        info = {}
        for ci in [c for c in dir(rs.camera_info) if not c.startswith("_")]:
            try:
                enum_val = getattr(rs.camera_info, ci)
                if isinstance(enum_val, int):
                    if device.supports(enum_val):
                        info[ci.lower()] = device.get_info(enum_val)
            except Exception:
                pass
        return info

    def profile(self):
        
        out = []
        for device in self._rs_ctx_devices:
            device_info = self._get_device_info(device)
            serial = device_info.get("serial number", "")
            if self._config['serial'] and serial != self._config['serial']:
                continue
                
            sensors_out = []
            for s in device.sensors:
                s_name = s.get_info(rs.camera_info.name) if s.supports(rs.camera_info.name) else "UnknownSensor"
                s_dict = {"name": s_name, "options": {}, "is_depth_sensor": False, "profiles": []}

                for opt_name in [o for o in dir(rs.option) if not o.startswith("_")]:
                    try:
                        opt = getattr(rs.option, opt_name)
                        if isinstance(opt, int) and s.supports(opt):
                            val = s.get_option(opt)
                            s_dict["options"][opt_name.lower()] = val
                    except Exception:
                        pass

                try:
                    depth_sensor = rs.depth_sensor(s)
                    s_dict["is_depth_sensor"] = True
                    s_dict["depth_scale"] = depth_sensor.get_depth_scale()
                except Exception:
                    pass

                try:
                    profiles = s.get_stream_profiles()
                    for idx, p in enumerate(profiles):
                        s_dict["profiles"].append(profile_to_dict(p))
                except Exception:
                    pass

                sensors_out.append(s_dict)

            out.append({
                "device": device_info,
                "sensors": sensors_out
            })
            
        if osp.exists(self._config['output_dir']):
            with open(osp.join(self._config['output_dir'], 'info.json'), 'w') as jf:
                json.dump(out, jf, indent=2, ensure_ascii=False)
        else:
            for i, entry in enumerate(out, 1):
                d = entry["device"]
                print(f"\n=== Device #{i} ===")
                print(f"  Name         : {d.get('name', 'Unknown')}")
                print(f"  Product Line : {d.get('product_line', 'Unknown')}")
                print(f"  Serial       : {d.get('serial_number', 'Unknown')}")
                print(f"  FW Version   : {d.get('firmware_version', 'Unknown')}")
                if "usb_type_descriptor" in d:
                    print(f"  USB Type     : {d.get('usb_type_descriptor')}")
                if "asic_serial_number" in d:
                    print(f"  ASIC Serial  : {d.get('asic_serial_number')}")

                for s in entry["sensors"]:
                    print(f"  - Sensor: {s['name']}")
                    if s.get("is_depth_sensor"):
                        print(f"      depth_scale: {s.get('depth_scale')}")
                    max_show = 6
                    for p in s["profiles"][:max_show]:
                        w = p.get("width", "-")
                        h = p.get("height", "-")
                        print(f"      profile: {p['stream']} {w}x{h} {p['fps']}fps {p['fmt']}")
                    if len(s["profiles"]) > max_show:
                        print(f"      ... (+{len(s['profiles']) - max_show} more profiles)")
            print("\n힌트: JSON으로 보고 싶으면 '--json' 옵션을 사용하세요.")

    def capture(self):
        def now_stamp():
            # 파일명용 타임스탬프 (UTC 기반, ns 단위 일부 포함)
            t = time.time()
            dt = datetime.utcfromtimestamp(t)
            return dt.strftime("%Y%m%dT%H%M%S") + f"_{int((t - int(t))*1e6):06d}Z"
        
        try:
            frames = self._rs_pipeline.wait_for_frames()
            ts = now_stamp()

            # 영상 프레임 정렬 (선택)
            if self._rs_align is not None:
                frames = self._rs_align.process(frames)

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            
            if self._config['color'] and cf:
                cimg = np.asanyarray(cf.get_data())  # BGR8
                if self._config['save_color']:
                    png_path = self._cdir / f"{ts}_{self._serial}.png"
                    cv2.imwrite(png_path.as_posix(), cimg)
                
                if self._config['save_color_npy']:
                    npy_path = self._cdir / f"{ts}_{self._serial}.npy"
                    np.save(npy_path, cimg)

            if self._config['depth'] and df:
                dimg = np.asanyarray(df.get_data())  # uint16 깊이 raw
                if self._config['save_depth']:
                    raw_png = self._ddir / f"{ts}_{self._serial}_raw16.png"
                    cv2.imwrite(raw_png.as_posix(), dimg)
                    dcolor = np.asanyarray(self._rs_colorizer.colorize(df).get_data())
                    cv2.imwrite((self._ddir / f"{ts}_{self._serial}_viz.png").as_posix(), dcolor)
                if self._config['save_depth_npy']:
                    raw_npy = self._ddir / f"{ts}_{self._serial}_raw16.npy"
                    np.save(raw_npy, dimg)
                
                if self._depth_scale is not None:
                    depth_m = dimg.astype(np.float32) * self._depth_scale
                    np.save(self._ddir / f"{ts}_{self._serial}_depth_m.npy", depth_m)
            
            if (self._config['pointcloud'] or self._config['save_ply']) and self._rs_pointcloud_generator:
                if cf:
                    self._rs_pointcloud_generator.map_to(cf)
                
                points = self._rs_pointcloud_generator.calculate(df)
                verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                colors = None
                if cf:
                    texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
                    c_img_data = np.asanyarray(cf.get_data())
                    
                    # 텍스처 좌표를 사용해 포인트별 색상 추출
                    tex_x = np.round(texcoords[:, 0] * c_img_data.shape[1]).astype(int)
                    tex_y = np.round(texcoords[:, 1] * c_img_data.shape[0]).astype(int)
                    colors = c_img_data[tex_y, tex_x]
                
                if self._config['pointcloud']:
                    np.savez_compressed(self._pcdir / f"{ts}_{self._serial}_pc.npz", points=verts, colors=colors)

                if self._config['save_ply'] and o3d:
                    # Open3D PointCloud 객체 생성
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(verts)
                    if colors is not None:
                        # RealSense의 BGR을 RGB로 변환하고 0-255를 0-1 스케일로 변환
                        colors_rgb = colors[:, [2, 1, 0]] / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

                    # PLY 파일로 저장
                    o3d.io.write_point_cloud(self._pcdir / f"{ts}_{self._serial}.ply", pcd, write_ascii=True)
                    print(f"[{ts}] Saved PLY file.")


            if self._config['infra1']:
                ir1 = frames.get_infrared_frame(1)
                if ir1:
                    ir1_img = np.asanyarray(ir1.get_data())  # uint8
                    ts = now_stamp()
                    if self._config['save_infra']:
                        cv2.imwrite((self._idir1 / f"{ts}_{self._serial}.png").as_posix(), ir1_img)
                    if self._config['save_infra_npy']:
                        np.save(self._idir1 / f"{ts}_{self._serial}.npy", ir1_img)

            if self._config['infra2']:
                ir2 = frames.get_infrared_frame(2)
                if ir2:
                    ir2_img = np.asanyarray(ir2.get_data())
                    ts = now_stamp()
                    if self._config['save_infra']:
                        cv2.imwrite((self._idir2 / f"{ts}_{self._serial}.png").as_posix(), ir2_img)
                    if self._config['save_infra_npy']:
                        np.save(self._idir2 / f"{ts}_{self._serial}.npy", ir2_img)

        except Exception as e:
            raise RuntimeError(f"[ERROR] There has error for capture: {e}")
            
    def stream(self):
        import signal
        stop = {"flag": False}
        def handle_sigint(sig, frame):
            stop["flag"] = True
            print("\n[INFO] 종료 신호 수신. 정리 중...")
            
        signal.signal(signal.SIGINT, handle_sigint)

        try:
            while not stop['flag']:
                self.capture()
        finally:
            self.finalize()
            
    def finalize(self):
        self._rs_pipeline.stop()
        print("[INFO] 종료. 저장 경로:", self._config['output_dir'])

        
if __name__ == '__main__':
    cam = IntelRealSense()
    cam.profile()
    # cam.capture()
    # cam.finalize()

    cam.stream()        
