# Rapor ve sunum kaynakları

> **Sınır notu (2026-08-24):** Bu klasördeki rapor ve konuşma metni E26 sonundaki tarihsel
> teslim anını korur; güncel runtime veya yeniden üretim kılavuzu değildir. E27'nin ilk
> “integrated” sonucu 2026-08-24'te calibration/evaluation sızıntısı nedeniyle yeniden
> ölçüldü, G1'i geçemedi ve servisten çıkarıldı. Güncel bilimsel sözleşme için kök
> `README.md`, `PLAN.md` H4–H6 ve `ml/EXPERIMENTS.md` son düzeltme kaydı esas alınmalıdır.

- `STAJ_RAPORU.md` — staj bitirme raporunun kaynak metni (Word kopyası Desktop'ta;
  `pandoc STAJ_RAPORU.md -o STAJ_RAPORU.docx --toc -M lang=tr` ile yeniden üretilir).
  Tek doldurulacak alan: kapaktaki `[KURUM ADI]`.
- `SUNUS_METNI.md` — 45 slaytlık desteye slayt-slayt konuşma metni + muhtemel sorular.
- Sunumun kendisi: `~/Desktop/PixelProof_Sunum_v2.pptx` (orijinal korunarak; 26 deney,
  E21–E26 finale bölümü, QA'den geçmiş).

Her sayının kaynağı `ml/EXPERIMENTS.md` günlüğüdür; rapor deney numarasıyla atıf verir.
