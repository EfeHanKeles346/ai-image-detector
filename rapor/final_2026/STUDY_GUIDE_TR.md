# PixelProof — Anlamak ve anlatmak için çalışma rehberi

Bu rehber teslim raporunun parçası değildir. İngilizce raporu ve sunumu öğrenmek için hazırlanmıştır. Raporun kendisi hocanın biçim kurallarını izler. Buradaki açıklamalar proje kaydını sadeleştirir; kişisel deneyimlerini sen doğrulamalısın.

## Önce şu fikri oturt

Gerçek bir mühendislik prototipi geliştirdik. Fotoğraf alabiliyor, eğitilmiş bileşenlerle analiz ediyor, bir sonuç ve gerektiğinde belirsizlik gösteriyor. Belirli geliştirme verilerinde gerçek fotoğraflara yapılan yanlış AI suçlamalarını ciddi biçimde azalttık. Ancak dünyanın bütün fotoğraflarında güvenilir çalıştığını kanıtlamadık. Bu iki cümle birlikte doğru.

Projenin hikâyesi yalnızca “model eğittik, skor yükseldi” değil. Başlangıçtaki iyi skorların başka kaynaklarda bozulduğunu gördük. Etiket ve ölçüm hatalarını düzelttik. Yeni modelleri eski hatalarına ve yeni oluşturdukları hatalara göre değerlendirdik. Bazı fikirler başarısız olduğu için sisteme alınmadı. Bunları kayıt altında tutmak projenin önemli bir çıktısı.

## Sunumu nasıl çalışmalısın?

Sunumda 13 anlatım slaytı ve 2 kaynak slaytı var. Önerilen süre toplam 13 dakika 15 saniye; bu gerçek prova ölçümü değildir. PowerPoint’in konuşmacı notlarında İngilizce anlatım bulunuyor. Önce Türkçe anlamını kendi cümlelerinle anlat, sonra İngilizce notları çalış. Notları kelimesi kelimesine ezberlemek yerine slaytın tek ana fikrini öğren.

| Slayt | Süre | Anlatacağın ana fikir |
|---|---:|---|
| 1 | 40 sn | Çalışan ama evrenselliği kanıtlanmamış araştırma prototipi |
| 2 | 50 sn | Bütün görüntüyü sınıflandırmak ile düzenlenen alanı bulmak farklı işler |
| 3 | 60 sn | Tanıdık veride iyi skor, yeni kaynakta başarının garantisi değil |
| 4 | 65 sn | Etiketler ve değerlendirme yöntemi yanlışsa sonuç da yanlış yorumlanır |
| 5 | 65 sn | Hazır görsel temsillerin üzerine projeye ait uyarlama ve uygulama kurduk |
| 6 | 75 sn | Aynı gerçek fotoğraflarda yanlış alarmlar 68’den 0’a / 14’e indi |
| 7 | 65 sn | 20 sayısal koşul geçti, ayrı koruma koşulu geçmedi |
| 8 | 65 sn | Model puanı olasılık değil; iki eşik ve iki görünüm kullanılıyor |
| 9 | 90 sn | Önceki kapsamlı final, galeri ve sonraki adaylar hâlâ hata gösteriyor |
| 10 | 65 sn | Model2 başka konumlara yeterince genellenemedi |
| 11 | 55 sn | Kod, demo, kayıtlar ve yazılım kontrolleri tamamlanan çıktılar |
| 12 | 55 sn | Başarı ile henüz kanıtlanmamış iddiaları ayrı tutuyoruz |
| 13 | 45 sn | Sonraki hedef bağımsız ve daha çeşitli değerlendirme |

## Modelin içini basitçe anlamak

Bir fotoğraf bilgisayarda sayılardan oluşur. DINOv2, CLIP ve DEAR gibi önceden eğitilmiş bileşenler bu görüntüden sayısal özellikler çıkarır. Bu, fotoğrafı tarif eden uzun bir sayı listesi gibi düşünülebilir. Fakat bu listenin her sayısına insanlar için tek bir anlam vermek doğru değildir.

Projemizde bu temsilleri kullanan bileşenler ve bir düzeltme modeli eğitildi. E92 bir deney/model sürümü adı. “92 puan” veya “92 katman” demek değil. Büyük temel modelleri sıfırdan eğittiğimizi söylememelisin. Projeye ait katkı; veriyi düzenlemek, uygun uyarlamaları eğitmek, ölçümü denetlemek ve çalışan uygulamaya bağlamak.

E92 eğitiminde 12.269 ana görüntü vardı: 7.674 gerçek, 4.595 AI. Her görüntünün dört işlenmiş görünümü 49.076 örnek görünümü oluşturdu. Sonraki araştırmalarda kullanılan 12.525 ana görüntü E92’nin eski eğitim sayısıyla karıştırılmamalı. Aynı fotoğrafın dört kopyası dört bağımsız fotoğraf değildir.

## En önemli sayılar ne anlatıyor?

Ana geliştirme karşılaştırmasında 160 gerçek ve 160 AI görüntü var. Aynı görüntüler hem orijinal hem de uzun kenarı en fazla 1080 piksel olacak şekilde küçültülüp JPEG75 ile kaydedilmiş durumda ölçüldü. Bu yüzden 640 görünüm, 640 bağımsız fotoğraf anlamına gelmiyor.

| Ölçüm | E43 orijinal | E92 orijinal | E43 işlenmiş | E92 işlenmiş |
|---|---:|---:|---:|---:|
| Gerçeğe yanlış AI uyarısı | 68/160 | 0/160 | 68/160 | 14/160 |
| AI yakalama | 156/160 | 159/160 | 154/160 | 159/160 |

159/160 yaklaşık %99,38 eder. Bu, o koşuldaki AI görüntülerini yakalama oranıdır; yüklenen herhangi bir fotoğrafta doğru olma ihtimali değildir. 14/160 = %8,75, gerçek görüntülerde yanlış alarm oranıdır. İkisinin paydaları farklı sınıfları temsil eder.

Sıfır yanlış alarm gözlenmesi, gelecekte hiç hata olmayacağı anlamına gelmez. Gerçek tarafta yalnızca on SIDD sahnesi var. Aynı sahneden gelen görüntüler bağımsız sayılmaz. Verilere tekrar tekrar bakıldığı için bu koleksiyon geliştirme verisidir. Sonradan ona “tamamen bağımsız final testi” adı veremeyiz.

## 20/20 meselesi

On sayısal koşul iki işleme durumu için kontrol edildi. Bunlar örneğin AI yakalama, gerçek yanlış alarm, dengeli doğruluk ve belirsizlik oranını içeriyor. Tam listesi raporun 9.1 bölümünde. Bir kısmı aynı sonuçtan türediği için birbirine bağlı; yirmi ayrı ve bağımsız veri kümesi değiller.

Ayrıca “E43’ün yakaladığı hiçbir AI fotoğrafını yeni model kaçırmasın” koşulu vardı. E92, E43’ün yakaladığı bir orijinal AI görüntüsünü kaçırdı. Toplamda daha fazla AI yakalasa da bu özel koşul başarısız. Dolayısıyla doğru ifade: “Yirmi sayısal geliştirme kontrolü geçti, fakat tam kabul sözleşmesi geçmedi.”

## Önceki büyük test ve E102 neden önemli?

E43 daha önce 1.000 gerçek ve 1.000 AI ana görüntülü E49-C testine girdi. Gerçeklerde 391 orijinale ve 490 işlenmiş kopyaya yanlış AI dedi; 20 koşulun 11’ini geçti. Bunlar E43’ün başka bir veri kümesindeki sonuçları. E92’nin 160 gerçek üzerindeki başarısını bu büyük finali geçti şeklinde anlatamayız: E92 gerekli kabul kapısını geçemediği için kendi E49 tekrar karşılaştırması açılmadı.

E102 ise aynı geliştirme verisinde işlenmiş gerçek yanlış alarmını 14’ten 12’ye indirdi ve E92’nin yakaladığı AI’ları korudu. Ama E43’e karşı kalan aynı bir AI kaybını çözemedi. Bu nedenle yeni aday servise alınmadı. E92 güncel demo sürümü; her tabloda en düşük hatayı veren son deney demek değil.

Korunan 300 AI MNW ve 100 gerçek HDR+ görüntüsü hâlâ skorlanmamıştı. Bunları toplamak kendiliğinden bağımsız ve dengeli bir final testi oluşturmaz. Kaynak, akrabalık ve kapsam denetimleri gerekiyor.

## Ekrandaki yüzde ve belirsizlik

Arayüzde puanın 100 ile çarpılması onu olasılık yapmaz. Örneğin 7,94 eşiği “%7,94 AI ihtimalinde suçluyoruz” anlamına gelmiyor. Bu modelin deneysel puan ölçeğinde seçilmiş çalışma sınırı. Üst eşik eski E48 kalibrasyonundan, alt eşik tüketilmiş E49 kalibrasyon/geliştirme incelemesinden geliyor; E92 için yeni ve bağımsız bir olasılık kalibrasyonu yapılmış değil. Bir başka modelin 50 puanı ile karşılaştırılamaz.

Orijinal görünüm üst eşik olan yaklaşık 7,94’e ulaşıyorsa AI sinyali görünür kalır. E92’nin iki görünümü de alt eşik olan yaklaşık 1,15’in altındaysa “belirgin AI izi bulunamadı” sonucu çıkar. Kalan durumlarda belirsizlik gerekir. Görünümler anlaşmazsa uyarı da gösterilebilir. Karar yuvarlanmış sayılara değil tam puanlara dayanır.

Eski E43 artık iki düşük E92 puanını otomatik olarak belirsize çevirmiyor; danışma niteliğinde ayrı bilgi. “İz bulamadım” yine de “kesin gerçek” değildir. Daha önce görülmemiş bir üretici de düşük puan üretebilir. Eşiği sırf bir fotoğrafta karar almak için keyfî ±5 oynatmak bu sorunu çözmez; başka fotoğraflardaki hataları gizleyebilir.

## Model2 neden bitmedi?

Model2 için bütün görüntüye AI demek yetmiyor; hangi piksellerin düzenlendiğini bulmak gerekiyor. On altı ana görüntülü, tek düzenleyicili pilotta düzenleme farklı konumlara taşınınca piksel sıralama başarısı düştü. İki konumla öğrenme bazı kayıpları düzeltti ama önceki konumdaki sonucu bozdu ve gerçek görüntülerde yanlış işaretlenen alanı artırdı.

Piksel AUC 0,742’den yeni konumda 0,566’ya düştü; yeni uyarlama bunu 0,638’e çıkardı. Ancak gerçek fotoğraflarda yanlış işaretlenen alan %17,11’den %25,74’e yükseldi. Bu nedenle aday reddedildi. Bu sayıların hiçbiri “fotoğrafların %74’ünde düzenlemeyi kesin bulduk” demek değildir.

## Hocanın sorabileceği sorular

**Sen ne yaptın, hazır model mi kullandın?** Hazır özellik çıkarıcılar kullanıldı. Projede veri denetimi, etiket eşleme, öğrenilen uyarlamalar, kontrollü değerlendirme, API ve web arayüzü birleştirildi. AI kodlama desteği kullanıldığını dürüstçe belirt. Kendi rolünü gerçekten yaptığın işler üzerinden anlat; tek başına bütün kodu elle yazdığını iddia etme.

**En büyük başarın ne?** Aynı geliştirme koleksiyonunda gerçek görüntülerde yanlış alarmları ciddi azaltırken yüksek toplam AI yakalamayı koruyan bir araştırma sistemi ve tekrar incelenebilir deney kaydı oluşturmak.

**En önemli hata neydi?** Bazı kaynaklarda sayısal etiket yönü tersmiş. Yanlış eşlemeyle elde edilen sonuçlara yapılan açıklamalar geçersizleşti. Eşleme açıklaştırıldı ve ilgili deneyler yeniden çalıştırıldı. Bir başka hata eşik seçiminin değerlendirme verisine bakabilmesiydi; ayrıştırıldı.

**Neden evrensel diyemiyorsun?** Sınırlı kamera/sahne/üretici kapsamı, tekrar kullanılan geliştirme verisi, çözülmemiş veri akrabalığı ve yeni kaynaklarda gözlenen hatalar var. Yaygın bir önceden eğitilmiş model kullanmak bu kanıt boşluğunu kapatmaz.

**1.245 test geçtiyse model doğru değil mi?** Bunlar yazılım testleri. İstek yönetimi veya hesaplama kuralı doğru çalışabilir; model yine yanlış tahmin yapabilir. Test sayısı görüntü doğruluk oranı değildir.

**Başarısız deneyleri neden anlatıyorsun?** Hangi varsayımların yanlış olduğunu ve neden bazı adayları kullanmadığımızı gösteriyor. Kaydı silmeden düzeltmek mühendislik güvenilirliğini artırır.

**Ürün olarak kullanılabilir mi?** Yerel öğrenci demosu olarak gösterilebilir. Evrensel doğruluk, kalibre edilmiş olasılık veya kurumsal üretim kullanımı kanıtlanmış değil. Şirket içinde devreye alındığını ya da ticari etki yarattığını söylemiyoruz.

**Daha fazla veri otomatik çözüm mü?** Hayır. Yeni verinin etiketleri, lisansı, kaynağı ve bağımsızlığı doğru olmalı; kopya veya aynı sahneden yeni görüntüler kanıtı düşündüğümüz kadar genişletmeyebilir. Veriyi neden seçtiğimizi DATASETS.md’de kayıt altına almak bu yüzden önemli.

## Son okuma ve prova

Önce raporun özeti, 4.6 sonuçlar ve 6. sonuç bölümünü oku. Sonra slayt 6–8’i kendi cümlelerinle anlat. Ardından 4.5 bölümündeki gelişim öyküsüne dön. Bilmediğin terimi ezberleyip geçme; örnek üzerinden anlamlandır.

Son bir provada saati açıp on üç ana slaytı anlat. On beş dakikayı geçersen başarısız deneylerin teknik ayrıntılarını kısalt; sayıların sınırlamalarını çıkartma. Kendi deneyimini, şirket birimini ve kişisel öğrenme paragraflarını tamamladıktan sonra rapordaki yer tutucuları kaldır. Canlı demoyu göstereceksen önceden aynı dosyalarla kontrol et; sonucu kesin gerçek/AI belgesi gibi sunma.

## Tarihçeden eklenen önemli ara aşamalar

E33–E41 sırasında yalnızca eşik değiştirmek veya kaynakları dengelemek farklı kamera
ve üreticilere güvenilir aktarım sağlamadı. E42'nin RR testi 16.953 ana görüntü ve
bunların 50.858 bağlantılı görünümünü içeriyordu; tüm kabul koşulları geçilmedi.
E43'ün içerikleri eşleştirilmiş rekonstrüksiyon testinde de ciddi zayıflığı vardı.
E44–E48'de uzman modelleri birleştirmek bazı AI türlerini yakalarken gerçek fotoğraf
hatalarını artırabildi; başka bir veri kümesinde GAN görüntüleri kaçırıldı. Bu yüzden
sonraki geliştirmelerde hem yanlış alarmı azaltmak hem önce yakalanan AI'ları korumak
ayrı koşullar olarak izlendi. Bu farklı testleri E92'nin 320 ana görüntülük geliştirme
karşılaştırmasıyla aynı başarı oranında birleştirme.
