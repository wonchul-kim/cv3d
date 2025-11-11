#!/usr/bin/env python3
import argparse, csv, sys, io
import numpy as np

def quat_to_rot(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError('Zero quaternion')
    x, y, z, w = q / n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

def sniff_delimiter(sample_bytes):
    sample = sample_bytes.decode('utf-8', errors='ignore')
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t'])
        return dialect.delimiter
    except Exception:
        # 기본은 쉼표
        return ','

def load_poses(csv_path):
    with open(csv_path, 'rb') as fb:
        raw = fb.read()
    delim = sniff_delimiter(raw)

    # BOM/공백 제거 후 텍스트 래핑
    txt = raw.decode('utf-8-sig', errors='ignore')
    f = io.StringIO(txt)

    # 먼저 DictReader로 시도
    reader = csv.DictReader(f, delimiter=delim)
    fieldnames = [fn.strip().lower() for fn in (reader.fieldnames or [])]

    def _to_float_list(row, cols):
        vals = []
        for c in cols:
            s = str(row[c]).strip()
            vals.append(float(s))
        return vals

    P, Rm = [], []

    if set(['x','y','z','qx','qy','qz','qw']).issubset(set(fieldnames)):
        # 정상 헤더 케이스
        for row in reader:
            row = {k.strip().lower(): v for k,v in row.items()}
            P.append(_to_float_list(row, ['x','y','z']))
            qx,qy,qz,qw = _to_float_list(row, ['qx','qy','qz','qw'])
            Rm.append(quat_to_rot(qx,qy,qz,qw))
    else:
        # 헤더가 없거나 컬럼명이 다름 → 일반 reader로 재파싱
        f.seek(0)
        simple = csv.reader(f, delimiter=delim)
        for r in simple:
            if not r or all((str(c).strip()=='' for c in r)):
                continue
            # 숫자가 아닌 첫 줄(가짜 헤더)이면 건너뛴다
            try:
                vals = [float(str(c).strip()) for c in r[:7]]
            except ValueError:
                # 이 줄은 헤더로 간주하고 스킵
                continue
            if len(vals) < 7:
                continue
            x,y,z,qx,qy,qz,qw = vals[:7]
            P.append([x,y,z])
            Rm.append(quat_to_rot(qx,qy,qz,qw))

    P = np.array(P, dtype=float)
    Rm = np.stack(Rm, axis=0) if len(Rm)>0 else np.zeros((0,3,3))
    if len(P) < 5:
        raise ValueError(f'Need at least 5 valid pose rows, got {len(P)}')
    return P, Rm

def solve_pivot(P, Rm, trim=0.0, iters=2):
    idx = np.arange(len(P))
    for _ in range(max(1, iters)):
        R1, p1 = Rm[idx[0]], P[idx[0]]
        A_list, b_list = [], []
        for i in idx[1:]:
            A_list.append(Rm[i] - R1)
            b_list.append(p1 - P[i])
        A = np.vstack(A_list)
        b = np.hstack(b_list)
        t, *_ = np.linalg.lstsq(A, b, rcond=None)
        p_star = p1 + R1 @ t
        errs = np.linalg.norm(P + (Rm @ t.reshape(3,1)).reshape(-1,3) - p_star, axis=1)
        if trim <= 0.0:
            return t, p_star, errs, idx
        keep = int(np.ceil((1.0 - trim) * len(idx)))
        idx = np.argsort(errs)[:keep]

    R1, p1 = Rm[idx[0]], P[idx[0]]
    A_list, b_list = [], []
    for i in idx[1:]:
        A_list.append(Rm[i] - R1); b_list.append(p1 - P[i])
    A = np.vstack(A_list); b = np.hstack(b_list)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_star = p1 + R1 @ t
    errs = np.linalg.norm(P + (Rm @ t.reshape(3,1)).reshape(-1,3) - p_star, axis=1)
    return t, p_star, errs, idx

def main():
    ap = argparse.ArgumentParser(description='Solve TCP (flange/currentTCP -> tool position) via pivot calibration.')
    ap.add_argument('--input', default='/wonchul/outputs/pivot_cal/pivot_poses_.csv', help='CSV path (with or without header)')
    ap.add_argument('--trim', type=float, default=0.15, help='Outlier trim ratio [0..0.4]')
    ap.add_argument('--tool-quat', type=float, nargs=4, default=[0,0,0,1], help='tool quaternion qx qy qz qw')
    ap.add_argument('--parent', default='flange_or_current_tcp', help='parent frame in TF output')
    ap.add_argument('--child',  default='vgc10_tcp',   help='child frame in TF output')
    args = ap.parse_args()

    P, Rm = load_poses(args.input)
    t, p_star, errs, used_idx = solve_pivot(P, Rm, trim=args.trim, iters=2)
    rms = float(np.sqrt(np.mean(errs[used_idx]**2)))
    mad = float(np.median(np.abs(errs[used_idx] - np.median(errs[used_idx]))))

    print('--- Pivot result ------------------------------------')
    print(f'poses total/used     : {len(P)} / {len(used_idx)} (trim={args.trim:.2f})')
    print(f't (parent->TCP) [m]  : {t[0]:.6f}  {t[1]:.6f}  {t[2]:.6f}')
    print(f'p* (pivot in base)   : {p_star[0]:.6f}  {p_star[1]:.6f}  {p_star[2]:.6f}')
    print(f'reprojection error   : rms={rms*1000:.2f} mm, median abs dev={mad*1000:.2f} mm')
    print('------------------------------------------------------')

    qx, qy, qz, qw = args.tool_quat
    cmd = (f'ros2 run tf2_ros static_transform_publisher '
           f'{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} '
           f'{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} '
           f'{args.parent} {args.child}')
    print('\n# Put this static TF into your launch (or run once for testing):')
    print(cmd)
    print('\n# Note: pivot solves position only. Set tool-quat to your bracket orientation.')
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
