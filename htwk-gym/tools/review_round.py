import json,glob,os,math,statistics as stt
def te(d):
    return math.hypot(d['pos_err_m']['median'], math.radians(d['heading_err_deg']['median']))*100
def rng(v):
    v=[x for x in v if x is not None]
    return (stt.median(v),min(v),max(v)) if v else (None,None,None)
roots=sorted(glob.glob('logs/force_ab/*'))
if not roots: print('force_ab 결과 없음')
else:
    R=roots[-1]; print('== force_ab:',R,'==')
    cells={}
    print('%-10s %-5s %4s %7s %7s %8s %6s %7s %8s %7s %7s'%('arm','mode','seed','pos cm','head°','과제cm','낙상','구간','/1000','events','FDF'))
    for p in sorted(glob.glob(os.path.join(R,'*','*_seed*','**','report.json'),recursive=True)):
        rel=os.path.relpath(p,R).split(os.sep); arm=rel[0]; mode,seed=rel[1].split('_seed')
        d=json.load(open(p)); de=d.get('disturbance_eval') or {}
        d['_ev']=de.get('events'); d['_fdf']=de.get('falls_during_force'); d['_as']=de.get('active_share')
        cells.setdefault((arm,mode),[]).append(d)
        print('%-10s %-5s %4s %7.2f %7.2f %8.2f %6s %7s %7.2f %7s %7s'%(
            arm,mode,seed,d['pos_err_m']['median']*100,d['heading_err_deg']['median'],te(d),
            d.get('falls'),d.get('segments_completed'),1000*(d.get('fall_rate_per_attempt') or 0),
            d['_ev'] if d['_ev'] is not None else '미기록', d['_fdf'] if d['_fdf'] is not None else '-'))
    print('\n-- seed 종합: median [min, max] --')
    print('%-10s %-5s %-20s %-20s %-16s %s'%('arm','mode','pos cm','과제오차 cm','낙상','events'))
    for (arm,mode),rs in sorted(cells.items()):
        pm=rng([r['pos_err_m']['median']*100 for r in rs]); tm=rng([te(r) for r in rs])
        fm=rng([r.get('falls') for r in rs]); ev=rng([r.get('_ev') for r in rs])
        print('%-10s %-5s %-20s %-20s %-16s %s'%(arm,mode,
            '%.2f [%.2f, %.2f]'%pm,'%.2f [%.2f, %.2f]'%tm,
            '%.0f [%.0f, %.0f]'%fm, int(ev[0]) if ev[0] is not None else '미기록'))
    print('\n-- clean -> force 변화 --')
    for arm in sorted({a for a,_ in cells}):
        c,f=cells.get((arm,'clean')),cells.get((arm,'force'))
        if c and f:
            print('  %-10s 과제오차 %.2f -> %.2f cm   낙상 %.0f -> %.0f'%(arm,
                stt.median([te(r) for r in c]),stt.median([te(r) for r in f]),
                stt.median([r.get('falls') or 0 for r in c]),stt.median([r.get('falls') or 0 for r in f])))
        if f and all((r.get('_ev') or 0)==0 for r in f):
            print('  !!! %s: force event 0회 — 강건성 근거 아님'%arm)
print('\n== I2b 평지 재평가 ==')
for p in sorted(glob.glob('logs/i2b_flat/**/report.json',recursive=True)):
    d=json.load(open(p))
    print('  바닥 %s | 외력 %s | pos %.2f cm | head %.2f° | 과제 %.2f cm | strict %.1f%% | 낙상 %s/%s'%(
        d.get('eval_terrain'),d.get('force_profile'),d['pos_err_m']['median']*100,
        d['heading_err_deg']['median'],te(d),d['success_rate_strict']*100,
        d.get('falls'),d.get('segments_completed')))
