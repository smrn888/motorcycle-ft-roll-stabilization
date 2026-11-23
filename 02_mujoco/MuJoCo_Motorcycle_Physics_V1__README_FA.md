# بسته اصلاح‌شده ضبط و بازپخش ویدئوی MuJoCo

این بسته برای ادامه همان پروژه است؛ فایل‌های تنظیم توقف‌گاه و Supervisor از نسخه‌های کدی که در گفتگو بارگذاری شده‌اند گرفته شده‌اند. بسته هیچ checkpoint یا مدل آموزش‌دیده‌ای ندارد و وزن‌های پروژه را تغییر نمی‌دهد.

## چرا نسخه قبلی خطا داد؟
نسخه محلی `mj_observed_env.py` آرگومان‌های `gimbal_limit` و `supervisor` را نمی‌شناخت، در حالی که `capture.py` برای B و C به آن‌ها نیاز دارد. نسخه قبلی ابتدا A را ضبط کرد و بعد هنگام ساخت B متوقف شد. در این بسته، بررسی سازگاری پیش از اجرای هر اپیزود انجام می‌شود.

## نصب محدود به پوشه مجاز MuJoCo
1. فایل‌های `capture.py`, `render.py`, `select_cases.py` را در پوشه `Three Models` قرار بده.
2. فقط این سه فایل را از پوشه `replacements_MuJoCo_Motorcycle_Physics_V1` در مسیر زیر جایگزین کن:
   `C:\Users\Moein\Desktop\Paper\RL-MPC\MuJoCo\MuJoCo_Motorcycle_Physics_V1`
   - `mj_observed_env.py`
   - `mj_dynamics.py`
   - `mj_build_model.py`
3. هیچ جست‌وجو یا تغییر فایلی در پوشه مادر `RL-MPC` لازم نیست. فایل‌های `Three Models` را تغییر نده، به‌جز کپی سه اسکریپت ویدئو.
4. قبل از جایگزینی، از همان سه فایل داخل پوشه مجاز یک نسخه پشتیبان محلی بگیر.

## اجرا
از PowerShell در پوشه `Three Models`:

```powershell
$MJ = "C:\Users\Moein\Desktop\Paper\RL-MPC\MuJoCo\MuJoCo_Motorcycle_Physics_V1"
python -m pip install mujoco imageio imageio-ffmpeg pillow numpy pandas

python .\capture.py --mj-dir "$MJ" --case healthy --seed 500000 --bucket 0 --configs A B --controllers pure residual --out video_evidence\healthy_500000_b0
python .\render.py --mj-dir "$MJ" --trace-dir video_evidence\healthy_500000_b0 --out video_evidence\healthy_comparison.mp4
```

برای مقایسه A و C، اجرا کن:
```powershell
python .\capture.py --mj-dir "$MJ" --case healthy --seed 500000 --bucket 0 --configs A C --controllers pure residual --out video_evidence\healthy_AC_500000_b0
python .\render.py --mj-dir "$MJ" --trace-dir video_evidence\healthy_AC_500000_b0 --config-order A C --out video_evidence\healthy_AC_comparison.mp4
```

## خرابی‌ها
از `--case` باید نام دقیق سناریو در `final_fault_benchmark.SCENARIOS` داده شود. برای دیدن نام‌ها:
```powershell
python -c "import final_fault_benchmark as f; print('\\n'.join(s[0] for s in f.SCENARIOS if s[0] != 'param_fault'))"
```
چهار سناریوی `param_fault` در wrapper فعلی پشتیبانی نمی‌شوند. ابتدا هر سناریو را با یک بذر/گروه ثابت ثبت کن، سپس همان سناریو را با A/B/C مقایسه کن.

## نکات صحت پژوهشی
- B یک محدودیت مفصلی واقعی در MJCF با حد ±0.8 رادیان است؛ فقط metadata نیست.
- C از Supervisor موجود در `MuJoCoDynamics` استفاده می‌کند؛ حدهای آن طبق پیاده‌سازی موجود 0.5/0.3 رادیان‌اند.
- `render.py` سیاست را دوباره اجرا نمی‌کند؛ فقط فایل NPZ ذخیره‌شده را بازپخش می‌کند.
- قبل از استفاده از ویدئو در مقاله، آمار JSON را با اجرای مرجع همان seed/scenario/bucket تطبیق بده.
- مدل بصری از هندسه‌های primitive استفاده می‌کند و تماس چرخ/زمین ندارد؛ آن را انیمیشن مسیر دینامیکی مدل ساده‌شده توصیف کن، نه ویدئوی واقع‌گرایانه موتورسیکلت کامل.
