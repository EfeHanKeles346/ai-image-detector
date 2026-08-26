# PixelProof — Yapay Zekâ Üretimi Görsel Tespiti

**Staj Bitirme Raporu**

**Hazırlayan:** Efe Han Keleş
**Kurum:** [KURUM ADI]
**Staj Dönemi:** Temmuz – Ağustos 2026
**Tarih:** 20 Ağustos 2026

---

## 1. Giriş

Bu rapor, Temmuz–Ağustos 2026 staj dönemimde geliştirdiğim PixelProof projesini anlatıyor:
bir fotoğrafın yapay zekâ tarafından üretilip üretilmediğini tespit etmeye çalışan, iki modüllü
bir araştırma sistemi. Rapor boyunca anlatılan her sayı, numaralı bir deneye (E1–E26) dayanıyor;
deneylerin tam kayıtları projenin deposundaki `ml/EXPERIMENTS.md` günlüğünde, karar tarihi ise
`HISTORY.md` dosyasında tutuluyor. Projeyi diğer staj çalışmalarından ayırdığını düşündüğüm
özellik, vardığı sonuçlardan çok bu sonuçlara nasıl varıldığı: her deney öncesinde hipotez
yazıldı, başarısız sonuçlar da kaydedildi, bir veri hatası bulunduğunda dört deney baştan
koşuldu ve hiçbir ölçüm, nasıl ölçüldüğü doğrulanmadan iddiaya dönüşmedi.

Raporun kalanı şöyle örgütlendi: 2. bölüm problemi ve neden zor olduğunu, 3. bölüm veriyi ve
metodolojiyi, 4–5. bölümler deneysel yolculuğu (temel modellerden büyük bulgulara ve itiraf
edilen büyük hataya), 6. bölüm projenin asıl katkısı olan karar katmanını, 7–8. bölümler ise
sınırları, gelecek işleri ve kazanımları anlatıyor.

## 2. Projenin Tanımı ve Problem

### 2.1 PixelProof nedir?

PixelProof, stajımızda geliştirdiğimiz ve bir görselin yapay zekâ (YZ) tarafından mı üretildiğine yoksa gerçek bir kamera fotoğrafı mı olduğuna karar vermeyi hedefleyen bir araştırma projesidir. Kod tabanının tamamında tek bir etiket kuralı kullandık: `1 = YZ üretimi`, `0 = gerçek`. Proje iki parçadan oluşuyor: makine öğrenmesi tarafı (`ml/`) config dosyalarıyla sürülen deneyleri, eğitim/değerlendirme kodunu ve dört eğitilmiş dedektörü barındırıyor; web tarafı (`app/`, Next.js) ise kullanıcının görsel yükleyip yöntem seçebildiği bir demo sunuyor. Demoda şu anda birbirine harmanlanmamış dört dedektör servis ediyoruz — harmanlamayı da denedik, ölçtük ve işe yaramadığını gördük: sekiz farklı birleştirme kuralının en iyisi tek başına ResNet'i yalnızca +0.002 AUC (Area Under the Curve — ROC eğrisi altında kalan alan) geçti, yani gürültü seviyesinde kaldı (HISTORY §2b.4).

| Yöntem | Ne yapar | En iyi olduğu bölge |
|---|---|---|
| SmallCNN | 32×32 CIFAKE taban çizgisi | çok küçük girdiler |
| ResNet-18 | doğal çözünürlüklü GenImage üzerinde ince ayar | <700px, sıkıştırılmış görseller |
| İstatistik | yeniden boyutlandırma olmadan tüm pikseller üzerinden 68 el yapımı öznitelik | orta boy, düşük dokulu görseller |
| Karo (tile) | 6×6 ızgarada doğal 128px kırpmalar, en yüksek 3 skorun ortalaması | ≥700px |

Projenin en güçlü modeli, 128px doğal karolar üzerinde ince ayar yapılmış ResNet-18'dir: Defactify üzerinde AUC 0.770 ve %61.4 YZ yakalama oranı (recall) ölçtük (E20). Buna rağmen bu modeli API'de bir "hüküm" olarak sunmuyoruz, çünkü %10 yanlış alarm (false positive) bütçesine göre oturttuğumuz eşik (threshold), eğitimde hiç görülmemiş en kötü kamera kaynağında %96 yanlış alarma çıkıyor (E20-v2). Stajın ortasında dürüst özet şuydu: sıralama problemi çözülmüş, karar verme problemi çözülmemişti. Bu düğümün nasıl çözüldüğü 6. bölümde anlatılıyor.

### 2.2 İki modül

- **Modül 1 — bu görsel YZ üretimi mi?** Yukarıdaki dört dedektörün cevapladığı ikili sınıflandırma problemi; stajın ana gövdesi.
- **Modül 2 — görselin neresi manipüle edildi?** Henüz inşa edilmedi, ama artık varsayımsal da değil: karo dedektörü zaten karo başına bir skor haritası üretiyor ve "hangi karo sentetik görünüyor" sorusu "görselin neresi kurcalandı" sorusuyla aynı. Ayrıca diskte, piksel seviyesinde referans maskeler içeren 13 adli veri setinden oluşan 78 GB'lık bir derleme tutuyoruz; böylece yerelleştirmeyi tartışarak değil ölçerek değerlendirebileceğiz (HISTORY §12, DATASETS.md).

### 2.3 Problemi zorlaştıran üç gerçek

**Birincisi, üreteçler sürekli değişiyor.** "YZ görseli" sabit bir sınıf değil, birkaç ayda bir genişleyen bir küme; "gerçek fotoğraf" ise sensör fiziğiyle sabitlenmiş durumda. Bu asimetriyi doğrudan ölçtük: aynı öznitelikler üzerinde yalnızca "gerçek" sınıfını öğrenen tek-sınıf model dağılım dışı (out-of-distribution, OOD) sette 0.688 AUC alırken, yalnızca "YZ" sınıfını öğrenen model 0.358 ile şansın altında kaldı (E8). Bugünün üreteçlerini ezberleyen bir model, yarın çıkan üreteci "eşleşme yok, demek ki otantik" diye damgalar — bir dezenformasyon dedektörü için olabilecek en kötü hata yönü. Genelleme maliyetini de sayıyla gördük: eski nesil üreteçlerde 0.888 olan AUC (E6), hiçbirinde eğitilmediğimiz beş modern üreteçli Defactify'da 0.760'a düştü (HISTORY §2b.1).

**İkincisi, her görsel işlenmiş halde geliyor.** İnternetteki her görsel en az bir kez JPEG ile sıkıştırılmış, çoğu yeniden boyutlandırılmış durumda; üretim artefaktları ise tam da bu işlemlerin sildiği yüksek frekanslarda yaşıyor. Bunu en acı şekilde kendi işleme hattımızda (pipeline) öğrendik: her görseli modele vermeden önce 224×224'e küçültüyorduk ve Defactify'daki üreteç-bazlı skorlar kaynak çözünürlükle neredeyse birebir hizalandı — en küçük görseller (270px) 0.896 alırken en büyükleri (1024px) 0.670'te kaldı; yani kanıtı model görmeden biz siliyorduk (HISTORY §2b.1, §4b). Mekanizmayı doğrulamak için aynı modele doğal çözünürlüklü yamalar verdiğimizde ayrım tam öngörülen yerde iyileşti ama gerçek fotoğraflardaki yanlış alarm %44'ten %96'ya fırladı — model eğitimde hiç keskin doğal piksel görmediği için keskinliğin kendisini "YZ" olarak okudu (HISTORY §2b.1). Ayrıca envanter notlarımızda açıkça kayıtlı bir boşluk var: bu dönemdeki ölçümlerimizin hiçbiri yeniden sıkıştırma altında yapılmadı (DATASETS.md, Gaps).

**Üçüncüsü, elimizde orijinal yok.** Klasik adli senaryonun aksine, karşımızdaki tek şey işlenmiş tek bir dosya; kıyaslayacağımız "temiz" bir referans, kamera negatifi ya da öncesi-sonrası çifti yok (bu lüks yalnızca Modül 2'nin laboratuvar veri setlerinde var — derlemedeki her manipüle görsel orijinaline işaret ediyor, gerçek hayatta ise etmiyor). Karar tamamen görselin kendi içindeki izlerden, körlemesine verilmek zorunda. Bunun sinsi bir sonucunu da ölçtük: dedektörlerimiz kısmen "üretilmiş görsel neye benzer" yerine "benim eğitim setimin gerçek fotoğrafları neye benzer" öğreniyordu; karo modeli gerçek fotoğrafların %79'una "YZ" diyordu ve gerçek sınıfını genişletmek, YZ yakalama oranından ödün vermeden AUC'yi 0.55'ten 0.884'e taşıdı (HISTORY §12b).

## 3. Veri ve Metodoloji

### 3.1 Veri envanteri

Çekirdek dört seti masaüstünde, 2026-07-28/29'da edindiğimiz 255 GB'lık arşivi ise `/Volumes/LaCie/pixelproof-datasets/` yolundaki harici SSD'de tutuyoruz (HISTORY §1, §1c).

| Set | İçerik | Rolü |
|---|---|---|
| CIFAKE (`archive`) | 100k eğitim + 20k test, 32×32; gerçekler CIFAR-10, sahteler Stable Diffusion | SmallCNN'in eğitim ve test seti |
| `archive1` | Yüksek çözünürlüklü YZ/gerçek görseller, kategori bazlı | Yalnızca OOD değerlendirme; **kirli olduğu E10'da kanıtlandı** (§3.2) |
| GenImage (`genimage_split`) | 9,917 eğitim / 1,742 test; gerçekler ImageNet, sahteler 7 üretece dengeli dağılmış | ResNet'in ve öznitelik modellerinin eğitim seti |
| Defactify (`defactify_test`) | 16,875 görsel: 2,851 gerçek MS-COCO + SD 2.1, SDXL, SD 3, DALL-E 3, Midjourney v6'dan ~2,800'er; iki sınıf da JPEG | Modern üreteç test seti — asla eğitimde kullanılmadı |

SSD arşivinin öne çıkanları: Modül 1 eğitimi için **CommunityForensics-Small** (260 GB'ın 47 GB'ı elimizde; 228 farklı üreteç modeli ve görsel başına üreteç meta verisi — üreteç çeşitliliği, üreteçler-arası genellemenin bilinen kaldıracı) ile **AI-vs-Real-balanced** (12 GB; 19,960 YZ / 19,660 gerçek, iki taraf da karışık formatlı, temiz); ek hacim olarak yalnızca karo modunda kullanılabilir iki set (AIGC-Detection-Benchmark, 30 GB; ai-vs-real-200k, 49 GB); test tarafında Defactify'a ek olarak Nano Banana Pro içeren `julienlucas` seti; Modül 2 için ise piksel maskeli 78 GB'lık manipülasyon derlemesi (DATASETS.md). Değerlendirme setlerini eğitimden ayırma gerekçemiz baştan beri aynı: model kendi eğitim dağılımına benzeyen veride çok iyi görünebilir; asıl soruyu — genelliyor mu, yoksa eğitim istatistiklerini mi ezberledi — ancak tamamen farklı kaynaktan gelen bir set cevaplar (HISTORY §1).

### 3.2 "Kullanmadan önce denetle" kuralının doğuşu: archive1 vakası

`archive1`, E1'den E6'ya kadar projenin OOD ölçütüydü ve kimse içine bakmamıştı. Şüpheyi sayılar doğurdu: öznitelik modeli bu sette tam 0.505 (yazı tura), lojistik regresyon ise 0.217 ile sistematik olarak ters skor alıyordu (HISTORY §2b.4). 2026-07-27'de seti denetlediğimizde (E10) tablo çarpıcıydı: 745 gerçek görselin %100'ü JPEG ve 138 farklı boyutta, 250 YZ görselinin %100'ü PNG ve tamamı 512×512 kare. Yalnızca genişlik/yükseklik/en-boy oranına bakan bir lojistik model sınıfları **AUC 1.000** ile ayırıyordu — set, içeriğe hiç bakmadan çözülebilen kusursuz bir kısayol taşıyordu (HISTORY §1b). CNN'lerimizin (Convolutional Neural Network, evrişimli sinir ağı) bu kısayoldan etkilenmediğini kontrollü deneylerle gösterdik — PNG'leri JPEG'e çevirip iki sınıfı da kare kırptığımızda performans +0.008 ile yukarı oynadı; mekanik neden, kod çözmenin dosya formatını ve `Resize`'ın boyut bilgisini zaten yok etmesiydi — bu yüzden E1'in %77.1'i ve E6'nın 0.888'i geçerli kaldı. Ama bu tasarım değil şanstı: aynı kısayol sondası, doğal piksel okuyan öznitelik modelinin görsel genişliğini %92.6 doğrulukla tahmin edebildiğini gösterdi (E8).

Bu vakadan iki kalıcı kural çıkardık. Birincisi: **her seti kullanmadan önce denetle ve hükmü kaydet.** Beş mekanik kontrol uyguluyoruz — sınıflar arasında dosya formatı farkı, şekil farkı, çözünürlük farkı, sıkıştırma (piksel başına bayt) farkı ve sınıf dengesi; `ml/tools/audit_datasets.py` bu beş kontrolü parquet/zip/tar içine açmadan bakarak çalıştırıp `DENETIM.md` üretiyor (HISTORY §1b). İkincisi ve daha incesi: **bir kusur diskalifiye sebebi değil, kullanım koşuludur.** Kısayol ancak model onu algılayabiliyorsa vardır; tüm görseli veren eğitim boyutları görür, 128×128 doğal karolarla eğitim ise göremez — karo hangi boyuttaki görselden gelirse gelsin modele hep 128×128 ulaşır. Bu yüzden "YZ tarafı %100 kare" diye işaretlenen bir set, tüm-görsel eğitimi için kullanılamaz ama karo eğitimi için tamamen güvenlidir; `DATASETS.md`'deki her kayıt bu ayrıma göre etiketlidir.

### 3.3 Metodoloji ilkeleri

- **Ön-kayıtlı hipotez.** Her numaralı deney, koşulmadan önce yazılmış bir hipotezle `ml/EXPERIMENTS.md`'ye giriyor: hipotez → config → sayılar → sonuç. Böylece "zaten öyle olacağını biliyorduk" tuzağına yer kalmıyor.
- **Negatif sonuçlar da kayıt altında.** Kaybeden fikirleri kazananlar kadar değerli sayıyoruz ve tartışmayla değil ölçümle kapatıyoruz: "yalnızca YZ'yi öğren" fikri 0.358 ile şansın altında kaldı (E8); sekiz kurallı harmanlama en iyi ihtimalle +0.002 getirdi ve doğruluğu artırmak yerine setler arasında taşıdığı için demoda iki skoru ayrı gösterip uyuşmazlığı bayraklamayı seçtik (HISTORY §2b.4).
- **Kontrollü karşılaştırma.** Öznitelik modelini ResNet'in kullandığı GenImage bölünmesinin birebir aynısıyla eğittik — aynı görseller, aynı bölünme, aynı test setleri; değişen tek şey yöntem. Aksi halde "öznitelik mi CNN mi" sorusunun cevabı bir anekdot olurdu (HISTORY §2b.2).
- **Config'le sürülen, tekrarlanabilir deneyler.** Kodda sihirli sayı yok: öğrenme oranını değiştirmek YAML düzenlemek demek, her checkpoint eğitildiği config'i içinde saklıyor ve eğitim/doğrulama bölünmesi sabit rastgele tohumla (seed) yapılıyor; rastgeleliğe duyarlı kritik ölçümleri tek tohuma bağlamayıp birden fazla (≥3) tohumla tekrarlamayı ilke edindik. Yeni bir mimari eklemek `MODEL_REGISTRY`'ye tek satırlık bir kayıt (HISTORY §2).
- **Ayrılmış test setleri.** Defactify ve `julienlucas` üzerinde asla eğitim yapmıyoruz; `archive1`'i yalnızca E1–E6 ile süreklilik için tutuyor, yeni iddialara dayanak yapmıyoruz (DATASETS.md).

## 4. Deneysel Yolculuk I — Temelden Kare-Kare Yönteme (E1–E11)

### 4.1 Başlangıç: temel CNN ve ilk uçurum (E1)

Projeye kasıtlı olarak basit bir modelle başladık: yaklaşık 300 bin parametrelik küçük bir CNN'i (Convolutional Neural Network — evrişimli sinir ağı) CIFAKE veri kümesinin 90 bin görsellik eğitim bölümünde eğittik. Sonuç kâğıt üzerinde parlaktı: doğrulama doğruluğu %96,8, ayrılmış test kümesinde %96,75 doğruluk ve 0,995 ROC-AUC (Receiver Operating Characteristic eğrisi altında kalan alan) ölçtük (E1). Ancak modeli hiç görmediği, yüksek çözünürlüklü 995 görsellik harici kümede denediğimizde doğruluk %77,1'e, AUC 0,800'e düştü (E1). Eğitim–doğrulama farkı yalnızca 1,2 puan olduğundan bunun klasik aşırı öğrenme olmadığını gördük; asıl darboğaz dağılım kaymasıydı — model 32×32 piksellik CIFAKE dünyasına özgü özellikler öğrenmişti. İçerideki %96,8 ile dışarıdaki %77,1 arasındaki bu uçurum, sonraki bütün deneylerin sorusunu tanımladı.

Ara deneyler bu teşhisi sağlamlaştırdı: aynı gömme vektörleri üzerinde eğittiğimiz dört klasik sınıflandırıcının tamamı CNN başlığının ±0,2 puan bandına yerleşti (E2) — yani darboğaz sınıflandırıcı değil, temsildi. Kümeleme analizi de aynı şeyi görselleştirdi: test kümesinde küme saflığı 0,965 iken harici kümede 0,749'a çöktü (E3). Veri ölçekleme deneyi ise doğruluğun log(veri) ile neredeyse doğrusal büyüdüğünü, ama asıl sorunun veri miktarı olmadığını gösterdi (E4).

### 4.2 Transfer öğrenmenin çöküşü ve ön işleme yasası (E5)

Doğal sonraki adım daha güçlü bir omurgaydı: ImageNet ön eğitimli ResNet-18'i CIFAKE üzerinde ince ayarladık. Dağılım içi sonuç yine yükseldi (%97,66 test doğruluğu), fakat harici kümede felaket yaşadık: doğruluk %25,2'ye, AUC 0,523'e — yani yazı-tura seviyesine — indi; model 995 görselin 984'üne "yapay" dedi (E5).

Bunun bir kapasite kaybı olmadığını bir kontrol deneyiyle kanıtladık: harici görselleri modele vermeden önce 32 piksele küçültüp tekrar 224'e büyüterek eğitim dağılımını taklit ettiğimizde doğruluk %25,2'den %72,0'a, AUC 0,523'ten 0,802'ye geri geldi (E5). Model bozulmamıştı; eğitimde hep bulanık 32→224 büyütmeleri görmüş, testte ise keskin doğal fotoğraflarla karşılaşmıştı. Buradan projenin en çok tekrar eden yasasını çıkardık:

> **Testte modele ne gösterilecekse, eğitimde de o gösterilmelidir.**

Aynı yasa üç ayrı deneyde üç ayrı kılıkta karşımıza çıktı: E5'te bulanık büyütme/keskin fotoğraf uyumsuzluğu, E6'da tersi (doğal çözünürlükte eğitilen modelin 32×32 CIFAKE'te %50'ye düşmesi), E7'de ise küçültülmüş kırpmalarla eğitilen modele doğal çözünürlüklü yamalar verilince gerçek fotoğraflardaki yanlış alarm (false positive) oranının %95,6'ya fırlaması. Üçü de model hatası gibi göründü, üçü de ön işleme (kaynak/işleme hattı) uyumsuzluğuydu.

### 4.3 Dönüm noktası: küçültme kanıtı siliyor (E7)

Doğal çözünürlüklü GenImage verisiyle eğittiğimiz ResNet (E6, archive1'de 0,888 AUC ile o güne kadarki en iyi sonuç) modern üreticilere karşı zorlanınca, Defactify kümesindeki beş görülmemiş üreticide bir stres testi yaptık (E7). Beklediğimiz genel düşüş gerçekleşti (0,888 → 0,760), ama asıl bulgu beklemediğimizdi — AUC'ler neredeyse mükemmel biçimde *kaynak çözünürlüğüne* göre sıralandı:

| Üretici | Kaynak (px) | AUC | Yapay yakalama oranı |
|---|---|---|---|
| DALL-E 3 | 270 | 0,896 | %93,7 |
| Midjourney v6 | 436 | 0,821 | %86,5 |
| SDXL | 1024 | 0,717 | %75,1 |
| SD 2.1 | 768 | 0,696 | %71,1 |
| SD 3 | 1024 | 0,670 | %68,7 |
| **Tümü** | | **0,760** | %79,0 |

Yönü not edelim: en *küçük* görseller en iyi yakalanıyor — bir çözünürlük kısayolu tam tersini üretirdi (E7). Mekanizma mekanikti: değerlendirme hattı her girdiyi 224×224'e küçültüyor; 1024² bir görsel 4,6 kat küçülürken 270² olan neredeyse hiç dokunulmuyor. Küçültme bir alçak geçiren filtredir, üretim izleri ise yüksek frekanslıdır. **Kanıtı, model daha görmeden kendi elimizle siliyorduk.** Doğal çözünürlüklü yama denemesi tam öngörülen yerlerde ayrımı iyileştirdi (SD 3 0,672→0,776; SDXL 0,725→0,800) ama gerçek fotoğraflarda yanlış alarm %43,8'den %95,6'ya patladı; toplam AUC yerinde saydı (0,764→0,776) — ön işleme yasasının üçüncü ihlali (E7).

### 4.4 Çözünürlükten bağımsız yol: 68 istatistik (E8)

Madem küçültme kanıtı siliyor, hiç küçültmeyen bir yöntem kurduk: her görselden, doğal çözünürlükte *her* piksel üzerinden hesaplanan 68 el yapımı istatistik çıkardık — kanal momentleri, kanallar arası korelasyon ve Bayer alt-örgü varyansı (CFA/demozaikleme izi — gerçek bir sensörün bıraktığı, üretici modellerin bırakmadığı iz), gürültü artığı istatistikleri, 16 bantlı radyal FFT (Fast Fourier Transform) spektrumu, yerel varyans yüzdelikleri, 8×8 JPEG blok izleri ve HSV istatistikleri (E8). Hepsi oran ya da piksel başına ortalama olduğundan vektör her boyutta aynı anlamı taşıyor.

Hipotez doğrulandı, hem de fazlasıyla: 128×128 doğal kırpma (crop128) modunda E7'nin çözünürlük sıralaması yalnızca kaybolmadı, *tersine döndü* — 1024 piksellik üreticiler en kolay hedef hâline geldi (SDXL 0,867'ye çıktı; CNN'in en kötü olduğu üç üreticide +0,09 ile +0,15 AUC fark attık), buna karşılık ağır sıkıştırılmış 270 piksellik DALL-E 3'te 0,377 ile şansın altına düştük (E8). Genel toplamda ise yöntem CNN'den geriydi (0,717'ye karşı 0,760): elimizdeki, bir **uzman**dı, bir halef değil. İki önemli dipnotu da kaydettik: tek-sınıf karşılaştırmasında "gerçek fotoğraf nedir"i öğrenen model archive1'de 0,688 ile denetimliyi (0,505) geçti (E8) ve bir kısayol sondası, 68 özellikten orijinal görsel genişliğinin crop128 modunda bile %92,6 doğrulukla tahmin edilebildiğini gösterdi (E8) — çözünürlük, boyuttan değil dokudan sızıyor.

### 4.5 Karışım denemesi: dürüst bir negatif sonuç (E9)

İki yöntem ayrık yerlerde başarısız olduğuna göre birleşimleri ikisini de geçmeliydi. Sekiz birleştirme kuralı denedik (ortalama, ağırlıklı, maksimum, sıra-normalize varyantlar). En iyi kural ResNet'i ortalamada yalnızca **+0,002** AUC geçti — gürültü (E9). Kazanç eklenmiyor, yer değiştiriyordu: Defactify +0,036 kazanırken archive1 −0,036 kaybetti; üretici bazında dördü düzelirken DALL-E 3 0,896'dan 0,685'e çöktü (E9). Ders net: sabit ağırlıklı bir karışım bir uzmanı kullanamaz; özellik modeli archive1'de neredeyse rastgeleyken (0,505) o sinyali iyi bir sinyale ortalamak, başka yerdeki kazancı yiyor. Karar olarak demoda iki skoru ortalamak yerine yan yana gösterip anlaşmazlığı işaretlemeyi seçtik (E9).

### 4.6 Kendi kıyaslama kümemizi denetlemek (E10)

E8'in archive1'deki tuhaf, sistematik *ters* skorları (lojistik regresyon 0,217) bizi altı deneydir kullandığımız kıyaslama kümesinin içine ilk kez bakmaya zorladı. Denetim sonucu ağırdı: gerçek sınıfın %100'ü JPEG ve dikdörtgen, yapay sınıfın %100'ü PNG ve 512×512 kare çıktı; yalnızca genişlik/yükseklik/en-boy oranı üzerinden kurulan bir lojistik model sınıfları **1,000 AUC** ile ayırıyordu (E10). Ama tek değişkenli kontrol deneyleri rahatlatıcı bir sonuç verdi: her iki kısayolu da kaldırdığımızda (ortak JPEG q90 kodlama + kare merkez kırpma) iki ağın AUC'si −düşmek yerine− +0,008 oynadı (E10). Sebep mekanik: yeniden boyutlandırma hattı konteyner formatını da boyutları da modele hiç göstermiyor. Yani E1'in %77,1'i ve E6'nın 0,888'i gerçek tespit performansı olarak geçerli kaldı; ama doğal piksel okuyan E8 modeli için aynı bağışıklık geçerli değildi (%92,6'lık sonda). Buradan taşınacak genel ders: **önyargılı bir veri kümesi ancak model o önyargıyı algılayabiliyorsa tehlikelidir** (E10).

### 4.7 Kare-kare yöntem: ölçek sorununun çözülüşü (E11)

Son adımın kıvılcımı demodaki tek bir görseldi: 1122×1402'lik ChatGPT üretimi bir görselde CNN %48 (yanlış), tam-görsel özellik modeli %94 (doğru), tek merkez kırpma %47 (yanlış) dedi (E11). Nedenini ölçtük: 128×128'lik merkez kırpma görselin piksellerinin %1,04'üydü ve merkeze denk gelen düz lacivert tişörtün gri-seviye standart sapması 0,027 iken tüm görselinki 0,283'tü — 10,6 kat daha düz bir yama (E11). Model dokusuz bir kumaş parçasına bakıp doğru cevabı vermişti: "bilmiyorum". Fikir sağlamdı; örnekleme kördü.

Çözüm, tek pencere yerine görseli doğal çözünürlüklü karelerden oluşan bir ızgarayla kaplamaktı. Yeniden eğitim gerekmedi: crop128 modeli zaten 128×128 doğal kırpmalarla eğitilmişti ve her kare tam olarak budur — ön işleme yasasının yeniden eğitim istemediği tek yer (E11). Izgara boyutunu ve birleştirme kuralını ayrı ayrı ölçtük:

| Izgara | En iyi kural | Ortalama AUC |
|---|---|---|
| 2×2 | üst yarı | 0,760 |
| 3×3 | top-3 | 0,799 |
| 4×4 | top-3 | 0,801 |
| 5×5 | top-3 | 0,807 |
| **6×6** | **top-3** | **0,821** |

En iyi 3 karenin ortalaması (top-3), düz ortalamayı her ızgara boyutunda geçti (6×6'da 0,821'e karşı 0,781), çünkü karelerin %21,4'ü doku tabanının altında kalıyor ve sıradan bir ortalamada ≈0,5 veren bu düz kareler kanıt taşıyanları boğuyor — tişört sorununun ölçeklenmiş hâli (E11). Üretici bazında sonuç projenin zirvesi oldu: SDXL'de **0,948 AUC** — modelin hiç eğitilmediği bir üreticide, E6'nın 0,888'lik rekorunun üzerinde; SD 3'te 0,894 (+0,224), SD 2.1'de 0,863 (+0,167); buna karşılık 436 piksellik Midjourney v6'da 0,580 ile CNN'e (0,821) yenildik (E11). Bu kesişimi de ölçtük: **~700 pikselin üzerinde kare-kare yöntem, altında CNN kazanıyor** — servis kodundaki uydurma 128px yönlendirme sabitini kanıta dayalı `TILE_RELIABLE_PX = 700` eşiğiyle değiştirdik (E11).

Böylece E5'ten beri süren ölçek sorunu hafifletilmedi, *çözüldü*: model her zaman 128×128 doğal piksel görüyor; çözünürlük yalnızca kaç kare çıkacağını değiştiriyor, bir karenin neye benzediğini asla (E11). İki yan kazanım da not düştük: görsel boyutları artık kısayol olamıyor ve kare başına skorlar, kurcalama yerelleştirmesinin (Modül 2) çekirdek mekanizmasını yan ürün olarak veriyor (E11). Bilinen sınır ise düşük çözünürlüklü, ağır sıkıştırılmış kaynaklar: DALL-E 3 (270px, ~16 KB) şansın altına düşüyor ve bu girdiler CNN'lerin alanında kalıyor (E11).

## 5. Deneysel Yolculuk II — Büyük Bulgular ve Büyük Hata (E12–E20)

### 5.1 E13: AUC iyi olabilir, karar yine de verilemez

E11'de karo (tile) yönteminin sıralama kalitesini ölçmüş ve SDXL üzerinde 0.948 AUC (Area Under the Curve — ROC eğrisi altında kalan alan) raporlamıştık. Elle test ederken gerçek bir fotoğrafın %99 "yapay zekâ" puanı aldığını görünce, bu sefer modelin çalışma noktasını — yani eşiğin (threshold) gerçekte nereye düştüğünü — doğrudan ölçtük (E13). Sonuç çarpıcıydı:

| Gerçek fotoğraf kaynağı | Yapay zekâ denen oran | Medyan puan |
|---|---|---|
| GenImage (eğitimde görülen) | %45.3 | 0.461 |
| Defactify (MS-COCO) | %93.3 | 0.935 |
| archive1 (Instagram) | %99.3 | 0.939 |
| **Toplam (900 görsel)** | **%79.3** | 0.887 |

Model, kendi eğitim setinin gerçek fotoğraflarını 0.461'de, hiç görmediği her gerçek kaynağı ise 0.93–0.94'te puanlıyordu (E13). Gerçek fotoğraflar 0.935'te, SDXL 0.993'te — arada yalnızca 0.06 var. Yanlış alarm (false positive) oranını %5'e çekmek için eşiği 0.992'ye ittiğimizde toplam yakalama oranı (recall) %27.3'e düştü; DALL-E 3 ve Midjourney'de sıfıra indi (E13). Buradan projenin kalıcı metodoloji kurallarından birini çıkardık: **AUC eşikten bağımsız bir sıralama ölçüsüdür; bir dedektörün iddiası AUC değil, çalışma noktası olmalıdır.** 0.948'lik AUC gerçek bir sıralama bilgisiydi ve aynı anda sahada kullanılamazdı.

### 5.2 E14: dar gerçek sınıfı — projenin en önemli bulgusu

E13'teki çöküşün nedenini önceden kaydettiğimiz bir hipotezle test ettik: sorun kalibrasyon değil, **dar negatif sınıf** olabilirdi. Gerçek yarısı tek kaynaktan gelen bir model "bu görüntü kamera izi taşıyor mu" sorusunu değil, "bu görüntü benim tanıdığım kaynağa benziyor mu" sorusunu öğrenir. Bunu izole etmek için beş kol kurduk: her kol tek bir kaynağın gerçek fotoğraflarıyla eğitildi, yapay zekâ yarısı (50,940 görsel) her kolda birebir aynı tutuldu ve gerçek bütçe 3,697'de eşitlendi — böylece değişken hacim değil, yalnızca *çeşitlilik* oldu (E14).

| Eğitim gerçek kaynağı | Kendi kaynağında yanlış alarm | Görülmeyen kaynaklarda |
|---|---|---|
| CommunityForensics | %0.3 | **%99.9** |
| GenImage | %23.7 | %91.7 |
| AIGC-Benchmark | %64.0 | %88.6 |

Tek kaynakla eğitilen her model, diğer kaynakların gerçek fotoğraflarının %88–99.9'unu "yapay zekâ" ilan etti. Beş kaynağı birleştirdiğimizde havuz AUC'si ~0.55–0.66 bandından **0.884'e** sıçradı ve bunun maliyeti sıfırdı: yapay zekâ yakalama oranı her kolda %99.5–100 kaldı (E14). Bu, projenin ölçtüğü en büyük etkiydi ve önceki her sonucu yeniden çerçeveledi: modeller "üretilmiş görüntü neye benzer"i değil, **"benim eğitim setimin gerçek fotoğrafları neye benzer"i** öğreniyor ve manifold dışındaki her şeye yapay zekâ diyordu. E12'de on kat verinin neden işe yaramadığını (hacim arttı, gerçek sınıf çeşitliliği artmadı), archive1'de kalibrasyonun neden çöktüğünü (Instagram işleme hattı görülmemiş bir kaynak/işleme hattı) tek seferde açıkladı. Doğru hedefin kaynaktan bağımsız fizik olan kamera izleri (PRNU, CFA korelasyonu, sıkıştırma geçmişi) olduğunu referans dokümanımız zaten söylüyordu; öncelik sıralamamızı buna göre değiştirdik: **gerçek sınıf çeşitliliği, her türlü omurga (backbone) yükseltmesinden önce gelir.** Dürüst bir dipnot: her koldaki 1:14 sınıf dengesizliği mutlak yanlış alarm oranlarını şişiriyordu; karşılaştırma bundan etkilenmez ve AUC eşikten bağımsız olduğu için 0.55→0.884 sıçraması ayakta kalır (E14). Dengeli tekrarında (E15) çeşitlilik tam tahmin edilen yerde işe yaradı — archive1 yanlış alarmı %30.1'den %19.8'e indi — ama Defactify'a hiç dokunmadı (0.717→0.692): dar gerçek sınıfı ve modern üreticileri ayırt etme zayıflığı iki ayrı problemdi (E15).

### 5.3 E19b: etiket hatası — %47 ters etiket ve dürüstlük vakası

Havuz denetimi sırasında (E19) projenin ürettiği en pahalı hatayı bulduk. Proje her yerde `0 = gerçek, 1 = yapay zekâ` kodlaması kullanıyor; beş eğitim kaynağından ikisi ise kendi HuggingFace üstverisinde **tam tersini** deklare ediyordu ve `build_pool.py` ham değeri olduğu gibi okuyordu. Sonuç: havuz indeksinin **%47.1'i (169,668 satırın 79,838'i) ters etiketliydi** (E19b). E12, E14, E15 ve E16'nın eğitildiği her model, yarı yarıya yanlış bir hedeften öğrenmişti.

Bu bölümü bir dürüstlük vakası olarak anlatmak istiyoruz, çünkü asıl ders sayılarda değil süreçte. Birincisi, üstveriye tek başına güvenmedik; yönü **üç bağımsız yolla** doğruladık: (1) 200 pikselin üzerindeki örnekleri gözle inceledik — ters kaynakların "0" etiketli her örneği açıkça difüzyon çıktısıydı; (2) üstverisi olmayan CommunityForensics'i `model_name` sütunu üzerinden çözdük — 0 etiketi %100 FFHQ (gerçek yüz fotoğrafı seti), 1 etiketi difüzyon model kimlikleri; (3) bir transfer sondası koştuk ve **sonuçsuz** çıktığını olduğu gibi raporladık — referans model o kaynaklara genellenemediği için (yine E14) bu test etiket yönünü çözemezdi ve onu kanıt gibi sunmak yanlış olurdu (E19b). İkincisi, indeksi yerinde yamalamak yerine sıfırdan yeniden kurduk; ham mı eşlenmiş mi etiket taşıdığı belirsiz bir CSV, çifte ters çevirme kazasına davetiyedir. Bozuk dosyayı da sildik değil, `pool_index_BOZUK_etiket.csv.bak` olarak sakladık. Üçüncüsü, kalıcı korkuluklar ekledik: `SOURCES` artık `label_map` taşıyor, `to_project_label()` tanımsız kaynakta hata fırlatıyor, `verify_labels()` sınıf sırası değişen bir yeniden dışa aktarımda çöküyor ve denetçiye altıncı bir kontrol eklendi (E19b).

Sonra kirlenen **dört deneyi aynı görüntüler, aynı tohumlar, aynı kodla — yalnızca etiket sütunu düzeltilmiş olarak — yeniden koştuk** (E19c):

| Deney | Karar |
|---|---|
| E14 — dar gerçek sınıfı | **Ayakta** (0.884 → 0.894; bulgu değişmedi) |
| E12 — on kat veri | Kısmen revize — sanılandan *daha çok* yardım etmiş (archive1 0.839 → 0.922) |
| E15 — dengeli havuz | Kısmen revize — sahaya en uygun model v3 değil v2 çıktı; %33'lük tıkanma değişmedi |
| E16 — donmuş DINOv2 | **Tersine döndü** |

E16 logdaki en büyük iddiaydı: DINOv2 sondası içerik-kontrollü Defactify'da 0.480 puan almış, biz de bunun üzerine "anlamsal kodlayıcılar yanlış yol, elle tasarlanmış istatistikler doğru aile" diye üç parçalı, gayet ikna edici bir çürütme yazmıştık. Etiketler düzelince aynı sonda Defactify'da **0.764** aldı — o tarihe kadarki en yüksek tam-görüntü AUC'miz — ve %10 yanlış alarm bütçesinde %40.4 yakalama oranıyla en iyi çalışma noktasını verdi (E19c). Yani tek bir bozuk etiket sütunu, doğru araştırma yönünün kendinden emin ve iyi gerekçelendirilmiş bir çürütmesini üretmişti; yazının inandırıcılığı hatayı görünmez kılan şeydi. Çıkardığımız ders yeni bir denetim kategorisi oldu: "bu veri seti yanlı mı?" ile "bu veri seti benim sandığım anlama mı geliyor?" farklı sorulardır ve o güne dek yalnızca ilkini soruyorduk. Ve bir veri düzeltmesinin akış aşağısındaki her şeyi yeniden koşmak opsiyonel değildir.

### 5.4 E20: üç model ailesi aynı karolarda — ImageNet ön-eğitimi kazanır

Havuz temizlenip karo geometrisi onarıldıktan sonra, kalan sınırın temsil olup olmadığını kontrollü biçimde test ettik: üç model ailesini **birebir aynı** 48,037 doğal 128px karo üzerinde, aynı kırpmalar, aynı tohum ve aynı uçtan-uca değerlendirmeyle karşılaştırdık — tek değişken model (E20).

| %10 yanlış alarm bütçesinde yakalama | 68 istatistik | **ResNet-18 @128** | SmallCNN @128 |
|---|---|---|---|
| Defactify (beş üretici) | %39.0 | **%55.5** | %30.5 |
| Midjourney | %5.0 | **%51.5** | %24.5 |
| DALL-E 3 | %4.5 | %9.5 | %4.0 |

**%55.5, projenin ürettiği en iyi çalışma noktasıydı** (E20). En bilgilendirici satır ise SmallCNN'in ikisine de kaybetmesi: sıfırdan eğitilen 0.3M parametrelik bir CNN (Convolutional Neural Network — evrişimli sinir ağı), aynı veride elle tasarlanmış fizikten bile kötü kaldı. Yani kazanan "istatistik yerine CNN" değil, **ImageNet ön-eğitiminin kendisi** (E20). İki şey ise düzelmedi: DALL-E 3 üç kolda da bozuk kaldı (%4.0–9.5, AUC şans seviyesinde ya da altında) — 270px ve ~16 KB'lık girdide karonun okuyacağı doku yok — ve eşik 0.5'te üç kol da gerçek fotoğrafların %86.5–94.6'sına yapay zekâ dedi (on adli veri setinden 2,314 otantik fotoğraf; E20). Darboğaz veriden temsile, temsilden **kalibrasyona** taşınmıştı. Ayrıca archive1'i değerlendirmeden bilerek çıkardık: gerçek/yapay yarıları arasındaki 7 katlık bayt/piksel farkı karolamadan sağ çıkıyor ve modelin üretim izi okumadan puan almasına izin veriyordu (E20).

### 5.5 E20-v2: kalibrasyon/değerlendirme ayrımı ve %96'lık en kötü kaynak

E13'ün dersini protokole gömdük: değerlendirme betiğini, Defactify'ın gerçek yarısını ve her üreticiyi **ayrık kalibrasyon/değerlendirme yarılarına** bölecek şekilde sertleştirdik — birleştirme kuralı ve eşik yalnızca kalibrasyon yarısını görüyor, tüm metrikler dokunulmamış yarıdan geliyor; her karonun puanı JSONL'e yazılıyor; beş birleştirme adayı yarışıyor ve seçilen eşik **değiştirilmeden** on adli gerçek kaynağa transfer ediliyor (E20-v2). Mevcut ResNet-18 kontrol noktasını yeni retrain'e para harcamadan bu protokolden geçirdik ve sonuç kararı değiştirdi: kalibrasyonda %10'a ayarlanan eşik, aynı kaynağın dokunulmamış yarısında bile %19 yanlış alarma kaydı; on adli kaynakta makro ortalama **%45.0**, **en kötü kaynakta %96.0** yanlış alarm ölçtük (E20-v2). Birleştirme kuralını değiştirmek (ortalama: %28.0 makro / %58.0 en kötü) yardım etti ama çözmedi; baskın neden değişken karo sayısı değil, kaynak/işleme hattı kaymasıydı (E20-v2). Havuzlanmış tek bir yanlış alarm sayısının tek bir kamera hattının felaket halinde çökmesini nasıl gizleyebildiğini burada gördük; makro ve en-kötü-kaynak sütunları artık başlık metriği. Karar da netti: bu kontrol noktasını API kararı olarak sunmuyoruz ve aynı başarısızlığı üç tohumla dokuz kez tekrarlamaya ödeme yapmadan önce donmuş harici taban çizgilerini (B-Free, CLIP) aynı protokolden geçireceğiz (E20-v2).

## 6. Karar Katmanı ve Sonuç Sistemi (E21–E26)

Faz 1 veri sorununu, E20 temsil sorununu çözmüştü; ama %10 yanlış alarm (false positive, FP) bütçesi için ayarladığımız eşik (threshold), görülmemiş en kötü kamera kaynağında %96 yanlış alarma fırlıyordu (E20-v2). Projenin son evresinde bu darboğazın bizim modele mi, yoksa görevin kendisine mi ait olduğunu ölçtük — ve çözümü temsilde değil, karar katmanında bulduk.

### 6.1 Dış sınav: en iyi hazır dedektörler de kapıda kaldı (E21, E21b)

"Belki bizim model yetersizdir" hipotezini, literatürün en güçlü iki hazır modelini kendi protokolümüzden geçirerek test ettik: 4.803 üreteçle eğitilmiş Community-Forensics ViT-S (ViT: Vision Transformer) ve içerik-hizalı eğitim yapan B-Free. Eşik yalnızca kalibrasyon yarısında seçildi, her şey el değmemiş yarılarda ölçüldü, aynı eşik on adli gerçek kaynağa taşındı (E21).

| metrik | tile ResNet-18 (bizim) | CF ViT-S | B-Free |
|---|---|---|---|
| Defactify AUC | 0.770 | 0.876 | **0.926** |
| Yakalama oranı (recall) | %61.4 | %70.8 | **%81.2** |
| Adli makro FP | %45.0 | %29.9 | **%23.6** |
| **En kötü kaynak FP** | %96.0 | %81.6 | **%96.8** |

(AUC: ROC eğrisi altındaki alan.) İki dış model bizi neredeyse her sütunda geçti; B-Free, bizim %10'da kaldığımız DALL-E 3 rotasını da büyük ölçüde çözdü (%68 yakalama, E21b). Ama ikisi de kapıda kaldı: en az bir görülmemiş kamera hattının gerçek fotoğraflarının %81'inden fazlasına "yapay" dediler. Üç bağımsız eğitim felsefesi aynı testte çöktü — kaynaklar-arası karar problemi bizim modelin değil, **görevin bir özelliği** (E21b); E14'ün öngördüğü buydu. Temsil alışverişi tek başına kapıyı açmıyor.

### 6.2 En-kötü-kaynak eşiği ve ret bandı: %96'dan %9.7'ye (E22, E22b)

Madem hiçbir temsil kapıyı geçemiyor, karar kuralını değiştirdik. E22'de üç kolun önbelleğe alınmış puanları üzerinde (tam koşu ~2 saniye) kaynağı-dışarıda-bırak (leave-one-source-out) protokolüyle ölçtük: eşiği tek kaynağa değil, on bir gerçek işleme hattının **en kötüsüne** göre seçmek, CF ViT-S'i projenin tarihindeki **kapıyı geçen ilk çalışma noktası** yaptı — görülmemiş hatlarda en kötü FP %6.6, %28.4 yakalama. Kendi tile modelimiz ise kalibrasyonla kurtarılamaz çıktı: kaynak-dayanıklı eşik altında %1.2 yakalama kalıyor; puanları kaynak-değişmez değil (bu sonuç, planlanan Stay-Positive denemesini de gereksizleştirdi).

Asıl kazanç ikinci mekanizmada: iki eşikli **ret bandı** (abstention band). Puan üst eşiğin üstündeyse "yapay", altındaysa "yetersiz kanıt". Her hat ailesi ~100 kalibrasyon görüntüsü verdiğinde B-Free bandı, on bir hattın hepsinde ≤%8 FP ile **%65.2 yakalamaya** ulaştı; %21.2 ret oranıyla (E22). Zehirli kaynak NIST2016'nın kafa karışıklığı %92 ret olarak yönlendirildi — yanlış suçlama yerine "kanıt yetersiz". E24'teki son yapılandırmayla bu zincir, E20-v2'nin %96'sını hiç yeniden eğitim yapmadan %9.7'ye indirdi.

Dürüstlük payı: E22b'de her manşet sayıya önyükleme (bootstrap) güven aralığı ekledik. Nokta tahminler bütçeyi geçiyor, %95 üst sınırlar geçmiyor (%13–16); ~50 görüntülük kalibrasyon yarılarıyla bu beklenen aritmetik. O günden beri kural: aralıksız manşet sayı rapora girmez.

### 6.3 Bandı dürüstleştiren üç politika (E23a–c)

**E23a — asimetrik bant.** Midjourney'nin %40'ının aktif olarak "gerçek" sayılması bir alt grup değil; üretecin bütün dağılımı B-Free uzayında gerçeklerin yanında oturuyor. Belirleyici bulgu başka: her kaçırma bütçesinde NIST2016'nın gerçek kapsamı %0 — "gerçek" hükmü hiçbir zaman tutarlı bir vaat olmadı ve yapay içeriğin aklanabileceği tek çıktıydı. Karar: bant asimetrik oldu; hükümler yalnızca "yapay" / "yetersiz kanıt". Yanlış-gerçek oranı %13.6'dan %0'a indi, yakalama ve FP'ye maliyeti sıfır.

**E23b — 2048 piksel sınırı.** NIST2016'nın sırrı 12 megapiksel: B-Free'nin beş 504px kırpması dev karenin ~%2'sini görüyordu. Uzun kenarı 2048'e indirmek, NIST'i kalibrasyondan tamamen çıkardığımız gerçek-görülmemiş senaryoda %35.2 → **%8.8 FP**'ye getirdi — bant artık on bir hattın hepsinde ~%65 yakalamayla kapıyı geçiyor. Takası da kayda geçirdik: E7'ye göre küçültme üretim izini siler; megapiksel-yapay yakalanabilirliğini, ölçülmüş en kötü hata modu olan megapiksel-gerçek korumasıyla bilerek değiştik.

**E23c — sıkıştırma bir eşik alanı.** Sosyal medya standardı bozunum (q50 + %75 küçültme) altında iki kol zıt yönde bozuluyor: CF güvenli tarafa düşüyor (FP %0'a iner), B-Free tehlikeli tarafa (NIST gerçeklerinin %41.3'ünü suçlar). Bozunmuş yarılarda eşik yenileme bütçeyi geri getiriyor ama B-Free yakalamanın üçte birini ödüyor (%65.2 → %42.8). Sonuç: servis sözleşmesine sıkıştırma-rejimi yönlendirmesi girdi; bayt/piksel zaten her istekte kayıtlı.

### 6.4 Reçetenin gerçek veriyle sınavı: sahibinin telefonu (E24)

E22'nin ürün vaadini ("yeni hat = ~100 kalibrasyon fotoğrafı + eşik yenileme, eğitim yok") en gerçekçi hatta test ettik: proje sahibinin iPhone'undan 207 kamera-orijinali fotoğraf (EXIF doğrulamalı, medyan 4032px — NIST'i zehirleyen sınıfın ta kendisi). CF donmuş eşikle %1.0 FP verdi; **sınırsız B-Free sahibinin kendi fotoğraflarının %38.2'sini suçlayacaktı.** 2048 sınırı bunu %12.6'ya, ~104 fotoğrafla tek eşik yenileme el değmemiş yarıyı **%9.7'ye — bütçe tutturuldu** — indirdi; yakalama maliyeti üç puan (%62.2). Reçete artık iki kez ölçüldü (E25'teki taze 2026 setinde donmuş eşikler %0.5/%6.0 FP verdi): **denetle → sınırla → ~100 kalibrasyon → eşiği yenile → bütçenin içindesin.** Ürün, bu cümledir.

### 6.5 OR kuralı ve tek hüküm (E26)

Canlı kullanım, kıyasların gizlediği bir tasarım hatasını yakaladı: ChatGPT üretimi bir yükleme CF'de eşiğin yedi katı puan aldı, ama tek-birincil tasarımda kör kol (B-Free'nin belgeli GPT körlüğü, E25) tek başına karar verip "yetersiz" dedi. Düzeltmeyi bütün önbellek setlerinde ölçtük: **herhangi bir kol kendi en-kötü-kaynak eşiğini aşarsa hüküm "yapay"** (OR kuralı). En kötü FP değişmedi (%9.7) — çünkü iki kolun yanlış alarmları farklı kaynaklarda yaşıyor (CF: Columbia; B-Free: iPhone) — buna karşılık FLUX %38→%64.5, Midjourney %7→%14, Nano Banana %29→%56.5, ve kaçan ChatGPT görseli yakalandı (E26).

Kapanış doğrulaması sahibinin öngörüsüyle geldi: 207 iPhone fotoğrafının tamamı iki sistemden geçirildi. Eski tile sinyali **207/207'sine "yapay"** dedi (medyan p 0.994 — E13'ün hastalığı en saf halinde); karar katmanı **21/207** işaretledi (%10.1 — tasarlanan bütçe). Arayüz de buna göre düzeltildi: ekranda artık tek hüküm var, araştırma sinyali "karara dahil değil" etiketiyle ve ölçülmüş FP oranıyla altta duruyor.

### 6.6 E31: temiz veri, DINOv2 ve bağımsız gerçek-foto kapısı

Staj sonrasında “en azından gerçekten eğitilmiş ve denenebilir yeni bir model” hedefiyle bağlı
LaCie arşivini baştan denetledik. 270,91 GB / 603.991 Parquet satırından hacmi değil çeşitliliği
seçtik; kaynak ve üreteç kotaları, grup-ayrık TRAIN/CALIBRATION rolleri ve kilitli testlerle exact
/ dHash sızıntı kontrolü kurduk. İki mekanik ret turundan sonra 5.650 yapay / 5.650 gerçek olmak
üzere 11.300/11.300 doğal 128px karo sıfır decode, boyut ve protected-overlap kaybıyla üretildi
(E31/B2).

Aynı sözleşmede E20, donmuş DINOv2 ViT-S/14 ve 68 adli özellik karşılaştırıldı. DINOv2 CALIBRATION'da
0,966 AUC, %90,72 güncel-üreteç macro recall, %4,67 macro ve %6,70 worst gerçek FP ile kazandı
(E31/B3). E20 ve fizik özellikleriyle max/stack birleşimleri gerçek tamamlayıcılık gösterdi; fakat
en iyi kazanç +3,05 puanda kaldı, ön-kayıt şartı +5'ti ve macro FP %5,34'e çıktı. Bu nedenle “çok
model tek model gibi çalışsın” önerisi ölçülerek reddedildi ve tek DINO adayı donduruldu (E31/B4).

Son kapı sonucu tersine çevirdi. Aday bağımsız 900-görünümlü MLLM DEVELOPMENT setinde GPT Image 2 ve
Nano Banana 2'yi yaklaşık %80,6 yakaladı, fakat gerçek görüntülerde macro FP **%83,63**, worst grup
%100 ve AUC **0,385** oldu (E31/B5). Gerçek bütçeyi sağlayan tanısal eşik AI recall'ı %0,33'e
indirdi; sorun kalibrasyon değil temsildi. Aday teknik olarak çalışır ve ayrı klasör CLI'siyle
denenebilir, ama servise alınmadı. DEVELOPMENT geçmediği için Qwen LOCKED FINAL'ın 40+40 satırı hiç
skorlanmadı. Bu fazın ana kazanımı bir başarı sayısı değil, temiz iç testin bile bağımsız gerçek
fotoğraf kapısının yerini tutmadığını gösteren uçtan uca bilimsel süreçtir.

## 7. Sınırlar ve Gelecek Çalışmalar

Sistemin sınırlarını sattığımız sayılar kadar net yazıyoruz:

- **GPT Image ailesi kör nokta.** Donmuş eşiklerde %6–8 yakalama, B-Free için sıralama düpedüz şans seviyesi (AUC 0.478, E25); OR kuralı bunu ancak %12'ye çıkarıyor (E26). Teselli, sistemin dürüst davranması: bu görüntülerin %92–94'ü sahte bir "gerçek" yerine "yetersiz kanıt" alıyor. Çözüm kalibrasyon değil, otoregresif/yerel-çokkipli aile üzerinde eğitilmiş bir temsil — yani veri edinme işi.
- **Güven aralıkları geniş.** Nokta tahminler %10 bütçenin altında ama %95 üst sınırlar %13–16 (E22b). Kaynak başına ~50 görüntülük kalibrasyon yarılarıyla bu kaçınılmaz; ilaç, hat başına kalibrasyon kütüphanesini büyütmek — E24'ün reçetesi zaten bunu üretiyor.
- **Sıkıştırma rejimi ayrı dünya.** Bozunmuş alanda bant %42.8 yakalamaya düşüyor (E23c); rejim yönlendirmesi tasarlandı ama tek bölmeyle ölçüldü, aralık kuralı burada da geçerli.
- **Midjourney zayıf halka kalıyor** (bant içinde %14, E26); B-Free uzayında bütün dağılımı gerçeklere komşu (E23a).
- **Modül 2 ("nerede düzenlendi?") park halinde:** kiremit yerelleştirme sinyali yalnızca difüzyon inpainting'de var (CocoGlide); klasik ekleme hattı kapandı. Gelecek hedef TGIF2.
- **Pratik notlar:** eşikler kol-bazlı puan ölçeğine özgü; B-Free lisansı kâr amacı gütmeyen kullanımla sınırlı, demoda bu yüzden ancak açık bayrakla yükleniyor.

Sıradaki işler, önem sırasıyla: kalibrasyon kütüphanesini yeni gerçek hatlarla büyütmek (aralıklar daralır), GPT-Image-sınıfı üreteç verisi toplamak, ret bandına konformal tahmin temelli istatistiksel bir taban denemek ve bozunmuş-alan bandını birden çok bölmeyle doğrulamak.

## 8. Sonuç ve Kazanımlar

Bu stajın dürüst özeti şu: **daha iyi bir dedektör yapmadık — dedektörleri kullanılabilir ve dürüst yapan katmanı yaptık.** Literatürün en iyi iki hazır modeli bile görülmemiş bir kamera hattında gerçek fotoğrafların %81–97'sini suçlarken (E21/E21b), aynı donmuş modellerin üstüne kurduğumuz karar katmanı — en-kötü-kaynak eşiği, asimetrik ret bandı, megapiksel sınırı, sıkıştırma yönlendirmesi ve OR kuralı — on iki hat üzerinde %9.7 en-kötü yanlış alarmla %62–65 yakalama verdi (E22–E24, E26). Tek cümlelik kanıt, proje sahibinin kendi telefonundan geldi: eski sinyal 207 gerçek fotoğrafın 207'sine "yapay" derken, son sistem tasarlanan bütçenin tam içinde 21'ini işaretledi (E26). Ve sistem bilmediği yerde susmayı biliyor: GPT-Image-sınıfı girdilere sahte bir güven yerine "yetersiz kanıt" diyor (E25).

Yöntem tarafında dört yetkinlik edindik. **Deney disiplini:** her deneyin hipotezlerini koşudan önce betiğe yazdık, eşiği hep kalibrasyon yarısında seçip el değmemiş yarıda ölçtük, rapora giren her eğitim sonucunu üç tohumla doğruladık (E20 eki: en kötü kaynak FP %86.2 ± 3.1 — çöküş tohum tesadüfü değil). **Veri denetimi:** her seti kullanmadan önce denetledik; E19b'de yazdığımız etiket-yönü koruması ilk yeni veri setinde gerçekten ateşledi (E25). **Dürüst ölçüm:** manşet sayılara güven aralığı iliştirdik, ortalamanın arkasına saklanmak yerine en kötü kaynağı manşet yaptık, her politikanın bedelini (E23b'nin megapiksel takası, E23c'nin yakalama faturası) kayda geçirdik. **Hata itirafı:** çürüyen hipotezleri (E23a'nın alt grup tahmini) bilgilendirici çürütme olarak yazdık, canlı kullanımda kaçan ChatGPT görselini gizlemek yerine E26'nın çıkış noktası yaptık ve kendi modelimizin kalibrasyonla kurtarılamayacağını ölçüp söyledik (E22). Bir dedektörün değeri en iyi gününde verdiği AUC değil, en kötü kaynağında verdiği sözdür — bu projeden aklımızda kalan cümle bu.

---

## Ek A — Deney Dizini (E1–E31)

| Deney | Tek satırla |
|---|---|
| E1 | Temel CNN: içeride %96.8, dışarıda %77.1 — genelleme farkı projenin konusu oldu |
| E2–E4 | Temsil darboğazı, gömme analizi, öğrenme eğrisi |
| E5 | Transfer öğrenme çöküşü: ön işleme uyumsuzluğu (%25.2) ve kontrol deneyiyle teşhisi |
| E6 | Doğal çözünürlüklü eğitim: 0.888 AUC, kalibrasyonun taşınmadığının keşfi |
| E7 | Dönüm noktası: küçültme kanıtı siliyor — skorlar kaynak çözünürlükle hizalı |
| E8 | 68 fizik-tabanlı istatistik: uzman model, tek-sınıf asimetrisi |
| E9 | Karışım denemesi: en iyi kural +0.002 — negatif sonuç, demo ayrık kaldı |
| E10 | archive1 denetimi: metadata kısayolu AUC 1.000; CNN'lerin mekanik bağışıklığı |
| E11 | Kare-kare yöntem: 6×6 + top-3, SDXL 0.948, 700px geçiş noktası |
| E12 | On kat veri: sıkıştırma uçurumu bulundu, hacim tek başına yetmedi |
| E13 | Çalışma noktası: AUC ≠ karar — gerçek fotoğrafların %79'una yanlış alarm |
| E14 | Dar gerçek sınıfı: projenin en önemli bulgusu (0.55 → 0.884) |
| E15–E16 | Dengeli çok-kaynaklı havuz; DINOv2 sonucunun etiket hatasıyla ilişkisi |
| E17–E18 | Modül 2 ölçümü: yalnız difüzyon-inpainting'de sinyal; ELA kapsam dersi |
| E19/E19b/E19c | Havuz hijyeni ve %47'lik etiket hatası: bulunması, düzeltilmesi, dört deneyin yeniden koşulması |
| E20 | Üç model aynı karelerde: ImageNet ön-eğitimi kazandı; v2 protokolü kaynak transferini ölçtü |
| E21/E21b | Dış sınav: dünyanın en iyi iki hazır dedektörü de kapıda kaldı — sorun görevin kendisinde |
| E22 | Karar katmanı: worst-source eşiği + ret bandı — %96 → %9.7 en kötü kaynak yanlış alarmı |
| E22b | Bootstrap güven aralıkları; sınırın varyansı da düzelttiğinin keşfi |
| E23a | "Gerçek" kararının kaldırılması: aklama kapısının kapatılması |
| E23b | 2048px megapiksel politikası: zehirli kaynağın kurtarılması |
| E23c | Sıkıştırma sütunu: eşikler sıkıştırma rejimine de bağlı |
| E24 | Sahibinin telefonu: eski model 207/207'ye karşı yeni katman 21/207 |
| E25 | 2026 üreteçleri probu: donmuş eşikler taze kaynakta tuttu; GPT ailesi kör nokta |
| E26 | OR kuralı: kör birincil kol gören kolu veto edemez; tek hüküm ekranı |
| E27–E30 | GPT-kol düzeltmesi ve güncel veri/test sistemi: MLLM DEVELOPMENT + sealed Qwen |
| E31 | 11.300 temiz karo, DINOv2 kazanımı, ensemble reddi ve bağımsız gerçek-foto falsifikasyonu |
