# بسته ضبط + رندر یک‌مرحله‌ای MuJoCo

این نسخه `capture.py` را به `render.py` وصل می‌کند: پس از ثبت مسیرهای واقعی کنترل‌کننده‌ها در NPZ، به‌طور خودکار MP4 مقایسه‌ای می‌سازد. هنگام رندر، سیاست دوباره اجرا نمی‌شود.

## فایل‌ها
- `capture.py`: اجرای اپیزودهای ثابت، ثبت trace و اجرای خودکار رندر.
- `render.py`: بازپخش هم‌زمان مسیرهای NPZ در یک ویدئوی مقایسه‌ای.
- `select_cases.py`: ابزار انتخاب/بررسی سناریوها.
- `replacements_MuJoCo_Motorcycle_Physics_V1/`: نسخه‌های سازگار فایل‌های محیط، دینامیک و ساخت مدل برای A/B/C.

## نصب
1. کپی `capture.py`, `render.py`, `select_cases.py` به پوشه `Three Models`.
2. اگر هنوز انجام نشده، از فایل‌های فعلی نسخه پشتیبان بگیر و سه فایل جایگزین را فقط در پوشه مجاز `MuJoCo_Motorcycle_Physics_V1` قرار بده.
3. پوشه‌های دیگر را جست‌وجو یا تغییر نده.

## اجرا: ضبط و رندر در یک دستور
```powershell
$MJ = "C:\Users\Moein\Desktop\Paper\RL-MPC\MuJoCo\MuJoCo_Motorcycle_Physics_V1"
python .\capture.py `
  --mj-dir "$MJ" `
  --case dropout_imu_p0.3 `
  --seed 500000 `
  --bucket 0 `
  --configs A B `
  --controllers pure residual `
  --out video_evidence\dropout_imu_500000_b0
```

پس از ثبت چهار trace، فایل MP4 به‌صورت خودکار در همان پوشه خروجی ساخته می‌شود. نام آن شامل سناریو، پیکربندی‌ها، کنترل‌کننده‌ها، بذر و bucket است.

برای فقط ضبط و بدون رندر، `--no-render` اضافه کن. اگر خروجی هم‌نام از قبل وجود دارد، از مسیر خروجی جدید استفاده کن یا فقط در صورت تمایل واقعی به بازنویسی، `--force` بده.

## سناریوهای بعدی
```powershell
python .\capture.py --mj-dir "$MJ" --case bias_phi --seed 500000 --bucket 0 --configs A B --controllers pure residual --out video_evidence\bias_phi_500000_b0
python .\capture.py --mj-dir "$MJ" --case actuator_saturation --seed 500000 --bucket 0 --configs A B --controllers pure residual --out video_evidence\actuator_saturation_500000_b0
```

## نکات
- چهار سناریوی `param_fault` توسط wrapper فعلی پشتیبانی نمی‌شوند.
- هشدار `cudart64_110.dll` معمولاً فقط یعنی CUDA/GPU در دسترس نیست؛ اگر برنامه ادامه دهد، این هشدار به‌تنهایی علت شکست نیست.
- هندسه بصری از مدل MJCF موجود استفاده می‌کند و تماس واقعی تایر و زمین در مدل نیست؛ در مقاله ویدئو را به‌عنوان بازپخش مدل دینامیکی ساده‌شده معرفی کن.
