# PixelProof — Staj Bitirme Sunumu Konuşma Metni
*(45 slayt · hedef ~20 dakika · konuşma dili)*

---

**SLAYT 1 — Açılış**
Herkese merhaba, ben Efe. İki aylık stajım boyunca tek bir soruyla uğraştım: elimizdeki bir fotoğraf, yapay zekâ tarafından mı üretilmiş, yoksa gerçekten bir kamerayla mı çekilmiş? Bu sunumda size Temmuz'dan bugüne yaptığım 26 deneyi anlatacağım — başarılarıyla, ve açıkçası epey bir başarısızlığıyla birlikte. Çünkü bu projenin asıl hikâyesi, işlerin ters gittiği yerlerde saklı.
[geçiş: önce problemin neden zor olduğuna bakalım]

**SLAYT 2 — Problem nedir?**
Bu soru ilk bakışta kolay görünüyor — "bir model eğitiriz, olur biter" diye düşünüyorsunuz. Ama üç şey işi bozuyor. Bir: üreteçler sürekli değişiyor; bugün çalışan bir dedektör, yeni bir model çıktığında sessizce çuvallıyor. İki: internetteki her görsel sıkıştırılmış ve yeniden boyutlandırılmış — yani aradığımız izler zaten kısmen silinmiş geliyor. Üç: elimizde karşılaştıracak bir orijinal yok; tek bir görsele bakıp karar vermek zorundayız.

**SLAYT 3 — Projenin iki modülü**
Proje iki modülden oluşuyor. Birincisi "bu görsel AI üretimi mi?" sorusuna tüm görsel üzerinden tek bir cevap veriyor. İkincisi "görselin neresi oynanmış?" diye bölge bölge inceleyip ısı haritası çıkarıyor. İkinci modülün ilk ölçümleri yapıldı ama şimdilik park hâlinde — bu sunumun ağırlığı birinci modülde olacak.

**SLAYT 4 — İlk model: kendi küçük CNN'imiz**
Hocalarımın da hep söylediği gibi, bir temel çizgi olmadan hiçbir iyileştirme ölçülemez — o yüzden işe kendi yazdığım küçük bir sinir ağıyla başladım. Üç yüz bin parametrelik, 32'ye 32 piksel girdi alan mütevazı bir model; yüz bin görselle eğittim. Kendi test setinde yüzde 96.8 aldı — harika görünüyor değil mi? Ama farklı bir veri setinde denediğimde yüzde 77.1'e düştü. Yirmi puanlık bu düşüş, aslında projenin geri kalanının ana sorusu oldu: bu model bir şey mi öğrendi, yoksa ezber mi yaptı?

**SLAYT 5 — Kod: küçük CNN'in yapı taşı**
Bu slaytta modelin yapı taşını görüyorsunuz — üç blok, aralarında küçültme, sonunda "AI olma olasılığı" diyen tek bir sayı. Detayına şimdi girmeyeceğim ama isterseniz soru kısmında satır satır konuşabiliriz.

**SLAYT 6 — İkinci deneme: transfer öğrenme**
Sonraki adım klasikti: madem küçük model yetmiyor, hazır ve güçlü bir model alalım. ResNet-18 milyonlarca fotoğrafla önceden eğitilmiş; kenar, doku, desen tanımayı zaten biliyor. Sonuç: kendi test setinde yüzde 97.7 — küçük CNN'den iyi. Ama farklı veri setinde yüzde 25.2. Bu bir düşüş değil, çöküş: 995 görselin 984'üne "AI" dedi. Daha güçlü model, daha kötü sonuç — burada bir terslik vardı.

**SLAYT 7 — Kod: transfer öğrenme nasıl yapılır**
Kod tarafı aslında çok basit: hazır ağın son katmanını atıp yerine kendi tek çıkışlı kafamızı takıyoruz. Merak ederseniz sonra detayına girebilirim; şimdi asıl soruya dönelim — bu çöküş neden oldu?

**SLAYT 8 — Sorun modelde değildi**
Bir kontrol deneyi cevabı verdi. Eğitim verimiz 32'ye 32'lik minicik görsellerdi ve onları büyüterek modele veriyorduk — yani model hayatı boyunca hep bulanık görseller gördü. Test ettiğimiz fotoğraflar ise net ve yüksek çözünürlüklüydü; model yabancı bulduğu her şeye "AI" dedi. Kanıt şu: aynı test görsellerini eğitimdeki gibi bulanıklaştırdığımda başarı yüzde 25'ten yüzde 72'ye fırladı. Buradan projenin merkez kuralı çıktı: modele testte ne gösterilecekse, eğitimde de tam olarak o gösterilmeli.

**SLAYT 9 — Projenin dönüm noktası**
Ve işte projenin dönüm noktası — benim için de stajın en çarpıcı anı. Yedinci deneyde üreteçlere göre başarıyı sıraladım ve tabloya baktığımda bir gariplik gördüm: DALL·E 3'ün 270 piksellik minicik görselleri 0.896 ile en iyi tespit edilirken, SD 3'ün 1024 piksellik kocaman görselleri 0.670'te sürünüyordu. Yani en küçük görseller en iyi yakalanıyordu. Neden? Çünkü biz her görseli modele vermeden önce 224'e 224'e küçültüyorduk — ve büyük bir görseli küçültmek, AI izlerinin yaşadığı ince dokuyu siliyor. Bir düşünün: aylardır "model neden bulamıyor" diye uğraşıyorduk. Ama kanıtı, model daha görmeden, biz kendi ellerimizle siliyorduk.
[geçiş: kısa bir duraklama — bu cümlenin oturmasına izin ver]

**SLAYT 10 — Kod: sorunun doğduğu yer**
İşte o silme işleminin koddaki yeri — üç satırlık, son derece standart bir görüntü işleme tarifi. Ama o tarif "kedi mi köpek mi" için yazılmış; kedi küçültmeye dayanır, üretim izi dayanmaz. Neden küçülttüğümüzü ve bu satırların hikâyesini isterseniz detaylı konuşabiliriz.

**SLAYT 11 — Peki nasıl ayırt edilir?**
Bu noktada geri çekilip temel bir soru sormam gerekti: bir AI görselini gerçek fotoğraftan ayıran şey tam olarak ne? Bunun cevabı görüntü adli bilimi denen bir alanda — bir görselin nasıl üretildiğini, bıraktığı izlerden geriye doğru okuma işi. Şimdiki üç slayt, dedektörün tam olarak neye baktığını anlatıyor.

**SLAYT 12 — Gerçek bir fotoğraf nasıl oluşur?**
Gerçek bir fotoğraf dört aşamadan geçiyor ve her aşama geriye bir iz bırakıyor. Sensörün üretim kusurları, o kameraya özgü sabit bir gürültü parmak izi bırakıyor — her pikselde. Renk filtresi her noktada tek renk ölçüyor, diğer ikisi komşulardan tahmin ediliyor — bu da renk kanalları arasında düzenli bir bağ yaratıyor. Sonra kameranın kendi işlemesi ve JPEG sıkıştırmasının 8'e 8'lik ızgara izi geliyor. Özetle: gerçek bir fotoğraf, fizikten gelen bir iz yığını taşıyor.

**SLAYT 13 — Yapay zekâ nasıl üretir?**
AI tarafında ise hikâye bambaşka. Difüzyon modelleri tamamen gürültülü bir kareyle başlıyor — televizyon karıncası gibi — ve metne bakarak o gürültüyü yüzlerce adımda temizleyip görüntüyü ortaya çıkarıyor. Kritik nokta şu: bu görüntü hiçbir zaman bir sensörden geçmedi. Kamera parmak izi yok, renk filtresi bağı yok — çünkü ortada hiç fotoğraf çekilmedi. Buna karşılık çözücü ağın kendi izi var, ve difüzyon ince dokuyu gerçek fotoğraflardan daha "düzgün" bırakıyor.

**SLAYT 14 — Peki biz neye bakıyoruz?**
Yani elimizde iki tür iz var: gerçek fotoğrafta olması gerekenler ve AI görselde bulunanlar. Burada benim en önemsediğim nokta şu: fizik izleri daha kıymetli. Çünkü yeni bir üreteç çıktığında "AI nasıl görünür" değişiyor — ama "kamera nasıl çalışır" değişmiyor.

**SLAYT 15 — Birinci yaklaşım: 68 ölçüm**
İlk yaklaşımım bu fizik bilgisini doğrudan koda çevirmekti: görselden, bu izleri arayan 68 sayı hesaplıyorum, sonra klasik bir sınıflandırıcı bu 68 sayıya bakıp karar veriyor. Renk kanalı bağı, gürültü kalıntısı, frekans dağılımı, sıkıştırma ızgarası, doku istatistikleri. Bu yolun en cazip tarafı: ölçümler görselin boyutundan bağımsız — 300 piksellik de olsa 4000 piksellik de olsa yine 68 sayı çıkıyor.

**SLAYT 16 — Kod: fizik izini ölçmek**
Bunlardan bir tanesinin kodu burada — renk filtresinin dört alt-ızgarasını ayrı ayrı ölçüp aralarındaki farka bakıyor; gerçek fotoğrafta bu dörtlü farklı davranır, AI görselde aynıdır. 68 ölçümün hepsi bu mantıkta; isterseniz detayına girebilirim.

**SLAYT 17 — Çözüm: kare kare inceleme**
Küçültme problemine bulduğum çözüm ise şuydu: küçültme yok. Görseli olduğu gibi bırakıyoruz, 128'e 128'lik parçalara bölüyoruz ve her parçayı ayrı ayrı puanlıyoruz — model her zaman orijinal piksel görüyor. Sonra en yüksek üç puanın ortalamasını alıyoruz ki gökyüzü, duvar gibi düz alanlar kanıtı boğmasın. Bunun güzelliği şu: görsel ne kadar büyük olursa olsun her parça 128'e 128 — çözünürlük sadece kaç parça çıktığını değiştiriyor, parçanın neye benzediğini değil.

**SLAYT 18 — Kod: görseli parçalara bölmek**
Parçalama kodunun kritik detayı şurada: eski hâli kenarlarda incelenmeyen bir çerçeve bırakıyordu — 500 piksellik görselde alanın yüzde 41'i hiç incelenmiyordu, yeni hâlde yüzde 0. İsterseniz yuvarlama detayına birlikte bakabiliriz.

**SLAYT 19 — Onlarca parçadan tek cevaba**
Peki onlarca parça puanını tek bir cevaba nasıl indiriyoruz? Üç kuralı ölçtüm: hepsinin ortalaması 0.781 verdi — çünkü parçaların yüzde 21'i dokusuz ve bunlar kanıtı boğuyor. Sadece en yüksek parçayı almak 0.802 — ama tek şanslı bir parça tüm karara hükmediyor, kırılgan. En yüksek üçün ortalaması 0.821 ile kazandı. Dürüst olmak gerekirse burada açık bir soru da bıraktım: "3" sabit bir sayı ama parça sayısı 16 ile 700 arasında değişiyor — bu oran hâlâ ayarlanmadı.

**SLAYT 20 — Kod: parçalardan tek cevaba**
Bu fonksiyon o kuralın kendisi — ve aynı zamanda iki iş yapıyor: tek sayı Modül 1'e, parça konumları Modül 2'nin ısı haritasına gidiyor. Detayını sorularda konuşabiliriz.

**SLAYT 21 — Kare hattının onarılması**
Bu onarımların toplam etkisi çarpıcıydı. Dokümanlarda "yüzde 100 kapsam" yazıyordu; oturup ölçtüğümde 12 megapiksellik bir fotoğrafta incelenen piksel oranı yüzde 4.8 çıktı. Onarım sonrası: yüzde 100 kapsam, kenar kaybı sıfır, üstüne 3.3 kat hız. Buradan aldığım ders basit: dokümana değil, ölçüme inanacaksın.

**SLAYT 22 — Veri setlerinin incelikleri**
Veri tarafında da öğrendiğim kritik bir şey var: bir veri seti, modele içeriğe hiç bakmadan cevabı verebiliyor. Eski test setimizde gerçek fotoğrafların hepsi JPEG ve dikdörtgen, AI görsellerin hepsi PNG ve kareydi — sadece en-boy oranına bakan bir model bu seti kusursuz ayırıyordu, tek pikseline bile bakmadan. O yüzden artık her veri seti kullanılmadan önce beş kontrolden geçiyor: format, şekil, çözünürlük, sıkıştırma, sınıf dengesi. Ve asıl incelik şu: bir kusur, seti çöpe atmayı gerektirmiyor — nasıl kullanılacağını belirliyor. "Tüm AI görseller 1024 kare" bilgisi tüm görsele bakan model için ölümcül bir kolaylık; ama 128'lik bir parçaya bakan model, parçanın hangi boyuttan geldiğini bilemez — aynı set biri için kullanılamaz, diğeri için tertemiz.

**SLAYT 23 — Bulunan kritik hata**
Şimdi size projenin en utandırıcı ama en öğretici anını anlatacağım — çünkü saklamak yerine anlatmak, bu projenin ilkesi oldu. Bir gün fark ettim ki iki veri seti etiketlerinde "sıfır eşittir AI" diyordu; biz "sıfır eşittir gerçek" sanıyorduk. Yani eğitim verimizin yüzde 47'sinde modele AI görsellerine "gerçek", gerçek fotoğraflara "AI" diye öğretmişiz. Dört deney bu zehirli veriyle koşulmuştu. Hepsini düzeltip yeniden koştum: biri ayakta kaldı, ikisi kısmen değişti — ve biri tamamen tersine döndü. "Güçlü bir sinir ağı bu işe yaramaz" diye rapora yazdığımız sonuç, aslında elimizdeki en iyi sonuçmuş. O gün öğrendim: yanlış veriyle alınan sonuç, sonuç değil.

**SLAYT 24 — Kod: %47 hatanın çözümü**
Çözümü de koda gömdük: her veri seti artık kendi etiket yönünü açıkça beyan etmek zorunda, beyan etmezse kod hata veriyor. Kural şu: sessizce yanlış cevap vermek, patlamaktan çok daha kötüdür. İsterseniz detayına girebilirim.

**SLAYT 25 — İlk zafer ve gerçeklik kontrolü**
Temiz veriyle ilk büyük sonuç geldi: SDXL üzerinde 0.948 sıralama başarısı — projenin o güne kadarki en iyisi. Model, bir gerçek ve bir AI görsel verildiğinde 100 seferin 95'inde doğru sıralıyordu. Ama aynı model gerçek fotoğrafların yüzde 79'una yanlışlıkla "AI" diyordu. Yani sıralayabiliyordu ama karar veremiyordu — bir doktorun "A hastası B'den daha hasta" demesiyle "herkesi ameliyat edelim" demesi arasındaki fark gibi. Buradan çıkan ders: bir iddia her zaman karar noktası olarak verilmeli, sadece sıralama skoru olarak değil.

**SLAYT 26 — Neden? Dar gerçek sınıfı**
Peki neden karar veremiyordu? Deney 14 bence projenin en önemli bulgusunu verdi: model "AI neye benzer"i öğrenmemişti — "benim eğitim setimin gerçek fotoğrafları neye benzer"i öğrenmişti, ve onun dışındaki her şeye "AI" diyordu. Bunu beş ayrı deneyle test ettim: gerçek fotoğrafları tek kaynaktan alınca skor 0.55 — neredeyse yazı tura. Beş kaynaktan alınca 0.884. Ve maliyeti sıfır: AI yakalama oranı hiç düşmedi. Ben buna "dar gerçek sınıfı hastalığı" diyorum — birazdan tekrar karşımıza çıkacak.

**SLAYT 27 — Son aşama**
Stajın son bölümünde her şeyi bir araya getirdim: dört faz. Altyapıdaki kayıp adımları düzelt, veri havuzunu temizle, kare hattını onar — ve en sonunda modeli değiştirmenin gerçekten işe yarayıp yaramadığını ölç.

**SLAYT 28 — Üç model, aynı veri, tek değişken**
Model yarışını bilimsel yapmaya çok dikkat ettim: aynı kareler, aynı veri, tek değişken model. Kazanan ImageNet ön-eğitimli ResNet-18 oldu — sıfırdan eğitilen küçük ağ, el yapımı fizik özelliklerini bile geçemedi. Buradan çıkan sonuç: darboğaz artık veride değil, temsildeymiş.

**SLAYT 29 — Kod: eğitim döngüsü**
Eğitim döngüsünün kritik kısmı dışarıda: her epoch sonunda doğrulama verisinde ölçüm yapıp en iyi epoch'a dönüyoruz — "kaç epoch" bir tahmin değil, ölçüm sonucu. Önceki sürümde sabit 3 epoch vardı ve arkasında hiçbir ölçüm yoktu; isterseniz detayını konuşabiliriz.

**SLAYT 30 — Sonuçları okumadan önce: kavramlar**
Şimdi sonuçlara geçmeden önce dört kavramı netleştireyim çünkü sonraki slaytlar bunlarla konuşuyor. Eşik: modelin puanının neresinden "AI" diyeceğimiz — 0.5 keyfi bir sayıdır, çizgiyi biz seçeriz. Yakalama: AI görsellerin yüzde kaçını bulduk. Yanlış alarm: gerçek fotoğrafların yüzde kaçına haksız yere "AI" dedik. Ve AUC sadece sıralamayı ölçer, kararı değil. En önemli nokta şu: eşiği sıfıra çekersen yakalama yüzde 100 olur — ama yanlış alarm da yüzde 100. Tek başına hiçbir sayı bir şey ifade etmez.

**SLAYT 31 — Karışıklık matrisi**
İşte gerçek sonuç: modelin hiç görmediği 600 test görseli — 500 AI, 100 gerçek. Model 500 AI görselin 307'sini yakaladı, 193'ünü kaçırdı. 100 gerçek fotoğrafın 19'unu haksız yere suçladı, 81'ini doğru bildi. Yakalama yüzde 61.4, yanlış alarm yüzde 19, kesinlik yüzde 94.2. Ama bence bu tablodaki en kritik kutu sağ üst değil, sol alt: bir haber fotoğrafını yanlışlıkla "sahte" ilan etmek, bir sahteyi kaçırmaktan çok daha pahalı bir hata.

**SLAYT 32 — Kod: eşiğin dürüstçe seçilmesi**
Bu kod eşiği nasıl dürüstçe seçtiğimizi gösteriyor: eşik kalibrasyon yarısından seçiliyor, sonuç modelin hiç görmediği diğer yarıdan ölçülüyor — cevabı görerek eşik seçmek, sınavdan önce soruları görmek gibidir. Detayını sorularda açabilirim.

**SLAYT 33 — Neden bu kadar zor?**
Bu grafik problemin özünü gösteriyor: yeşil gerçek, kırmızı AI — ve iki tepe ayrı değil, iç içe. Bu yüzden hangi eşiği seçersen seç bir bedel ödüyorsun. Eşik 0.989 gibi uç bir noktaya itilmek zorunda kaldı, çünkü model neredeyse her şeye yüksek puan veriyor.

**SLAYT 34 — Eşik seçimi bir takas**
Yani eşik seçimi bir takas: düşürürsen daha çok AI yakalarsın ama daha çok masum fotoğrafı suçlarsın; yükseltirsen masumları korursun ama sahtelerin çoğu elini kolunu sallayarak geçer. Bu yüzden bütün karşılaştırmaları sabit bir bedelde yapıyoruz: "yüzde 10 yanlış alarma izin verirsem kaç AI yakalarım?"

**SLAYT 35 — Asıl problem burada görünüyor**
Ve asıl problem bu slaytta ortaya çıkıyor. Aynı eşiği 10 farklı kamera kaynağından gelen gerçek fotoğraflara uyguladım — yüzde 10 için ayarlanan yanlış alarm, bir kaynakta yüzde 96'ya fırlıyor. Yani o kameradan gelen her 100 gerçek fotoğrafın 96'sına "AI" diyoruz. Bu, Deney 14'teki dar gerçek sınıfı hastalığının ta kendisi: model AI'ı tanımıyor, kendi eğitim setinin fotoğraflarını tanıyor.

**SLAYT 36 — Ama hâlâ kullanılamıyor**
Özetle iki yüzlü bir tablo: sıralama tarafında yüzde 61.4 yakalama — projenin en iyi değeri. Karar tarafında en kötü kaynakta yüzde 96 yanlış alarm. Ve bu bir kalibrasyon ayarıyla düzelecek bir şey değil — kaynak kayması denen ayrı bir problem, ayrı bir çalışma gerektiriyor.

**SLAYT 37 — Deney 20 sonunda durum**
Deney 20'nin sonunda durum dürüstçe şuydu: veri temiz, girdi işleme tam, model seçimi ölçülmüş — üç tik. Ama karar verme çözülmemiş, küçük ve sıkıştırılmış görseller üç modelde de çalışmıyor — iki çarpı. Yani ölçtüğümüz her şey sağlamdı ama ortada kullanılabilir bir ürün henüz yoktu.
[geçiş: ve burada stajın son perdesi başlıyor]

**SLAYT 38 — Son perde**
Son iki haftada kendime şu soruyu sordum: madem bizim model karar veremiyor, dünyanın en iyileri verebiliyor mu? Deney 21'den 26'ya uzanan bu son perde, projenin en ilginç cevabını verdi — kapıyı geçen, model değil, karar katmanı oldu.

**SLAYT 39 — E21: Dünyanın en iyileri bizim sınavda**
Alandaki en iyi iki hazır dedektörü indirdim: 4.803 üreteçle eğitilmiş Community-Forensics ve CVPR 2025'ten B-Free. İkisi de bizim modelimizi her kolonda geçti — en iyisi 0.926 AUC'ye ulaştı. Ama ikisi de aynı kapıda kaldı: görmedikleri kameranın gerçek fotoğraflarını suçladılar — en kötü kaynakta yüzde 96.8 yanlış alarm. Üç farklı eğitim felsefesi, aynı başarısızlık. Demek ki sorun modelde değil, görevin kendisindeydi.

**SLAYT 40 — E22: Çözüm karar kuralındaymış**
Çözüm ise şaşırtıcı derecede ucuz çıktı: yeniden eğitim yok, sadece eşik kuralı değişti. Eşiği tek kaynağa göre değil, 12 gerçek kamera kaynağının en kötüsüne göre kuruyoruz. Ve karar iki seçenekli: "AI tespit edildi" ya da "yeterli kanıt yok" — sistem asla "gerçektir" demiyor. Sonuç: en kötü kaynaktaki yanlış alarm yüzde 96'dan yüzde 9.7'ye indi, yüzde 65 AI yakalamayla — ve bütçe hiçbir kaynakta aşılmadı. Yeni bir kamera eklemek için gereken tek şey yaklaşık 100 kalibrasyon fotoğrafı. 26 deneyin 22.'sinde, projenin ilk kullanılabilir çalışma noktası buydu.

**SLAYT 41 — E23: Kararın etrafındaki üç koruma**
Sonra bu kararın etrafına, her biri ölçülmüş bir zaafiyeti kapatan korumalar ördüm. "Gerçektir" kararını tamamen kaldırdık — çünkü hiçbir kaynakta tutarlı verilemediğini ölçtük ve AI içerik o kapıdan aklanabilirdi. 2048 piksel sınırı geldi — 12 megapiksellik fotoğraflar her modeli şaşırtıyordu; zehirli kaynak yüzde 35'ten yüzde 8.8'e düştü. Ağır sıkıştırmada yakalamanın düştüğünü sistem saklamıyor, kullanıcıya rozet olarak söylüyor. Ve her manşet sayının yüzde 95 güven aralığı raporlanıyor. Ortak tema şu: dürüstlük bir tasarım kararı — sistem kendi sınırlarını kullanıcıya kendisi söylüyor.

**SLAYT 42 — E24: En zor test — sahibinin telefonu**
Ve en zor test: kendi telefonum. 207 iPhone fotoğrafımı, sistemin hiç görmediği 12. kamera hattı olarak verdim. Eski modelimiz bu fotoğrafların hepsine "AI" dedi — 207'de 207, üstelik medyan güveni 0.994'tü. Yani model, benim çektiğim her gerçek fotoğrafa neredeyse tam emin şekilde "sahte" diyordu — dar gerçek sınıfı hastalığının en saf hâli. Yeni katman, aynı 207 fotoğrafta sadece 21'ini işaretledi — tam da tasarladığımız yüzde 10 bütçenin içinde. O an şunu hissettim: iki aydır kurduğumuz her şey — denetle, sınırla, 100 fotoğrafla kalibre et — gerçek hayatta, gerçek bir telefonda çalıştı. Ve bu tarif artık iki kez doğrulanmış durumda.

**SLAYT 43 — E26: Tek hüküm ve canlı demo**
Son dokunuş bir hüküm kuralıydı: iki dedektörden biri eşiğini aşarsa karar "AI" — ve yanlış alarm bütçesi bozulmadan, yüzde 9.7'de. Bu kural gerçek bir vakayla bulundu: bir ChatGPT görselini bir kol 7 kat farkla yakalıyordu ama kör olan "birincil" kol kararı veto ediyordu. Artık ekranda tek karar var — kim verdi, hangi eşikle, yazıyor. Ve demo localhost'ta canlı: isterseniz sunumdan sonra bir görsel yükleyip kararı ve gerekçesini birlikte görebiliriz.

**SLAYT 44 — Açık kalanlar**
Bitirmeden, neyin çözülmediğini de söylemem lazım — çünkü hepsi ölçüldü, hiçbiri saklanmıyor. Bir: GPT Image ailesi her modelin kör noktası — OR kuralıyla bile yüzde 12 yakalama; aday çözüm GPT verisiyle eğitilmiş üçüncü bir kol, kabul şartı hazır: bütçeyi bozmayacak. İki: güven aralıkları geniş — kaynak başına 50 civarı kalibrasyon görseli var, daha fazlası lazım. Üç: Modül 2 park hâlinde — bölge tespiti şimdilik sadece difüzyon-inpainting'de sinyal veriyor.

**SLAYT 44B — E31: çalışan aday neden servise girmedi?**
Sonraki turda harddiskteki 270 gigabaytı denetledik, testlerle çakışmayan 11.300 dengeli karo seçtik ve DINOv2 adayını eğittik. İç kalibrasyonda güncel AI yakalama yüzde 90,7 ve gerçek yanlış alarm yüzde 4,7 idi; eski modeli geçti. İki modeli bağlamak da +3 puan kazandırdı ama önceden koyduğumuz +5 şartını geçmedi, o yüzden tek modeli seçtik. Sonra bağımsız gerçek-foto sınavı geldi: AI yakalama yüzde 80,7 kaldı ama gerçek fotoğrafların yüzde 83,6'sına AI dedi, AUC 0,385'e ters döndü. Eşiği yükseltince AI yakalama yüzde 0,33'e indi. Yani model teknik olarak çalışıyor ama ürün olarak güvenli değil. Servise koymadık ve kilitli son testi açmadık. Bu slayt, test sistemimizin sahte bir başarıyı gerçekten durdurduğunun kanıtı.

**SLAYT 45 — Bu projenin asıl çıktısı**
Kapanışta size iki şey söylemek istiyorum. Birincisi: bu iki ayda daha iyi bir dedektör yapmadık — dünyanın en iyileri bile bizim sınavda aynı kapıda kaldı. Biz, dedektörleri kullanılabilir ve dürüst yapan katmanı yaptık: denetlenen veri, ölçülen sınırlar, "yeterli kanıt yok" diyebilen bir karar. İkincisi ve bence asıl çıktı, bir yöntem: her deney öncesi hipotez yazıldı, 26 deneyin 8'i başarısızdı ve hepsi kayıtlı, bir veri hatası bulununca 4 deney geri alınıp yeniden koşuldu. Bu stajdan en kalıcı öğrendiğim cümleyle bitireyim: bir ölçüm ne kadar iyi görünürse görünsün, nasıl ölçüldüğü doğrulanmadan iddia olmaz. Teşekkür ederim — sorularınızı memnuniyetle alırım.

---

## MUHTEMEL SORULAR

**1. Küçültmenin kanıtı sildiğini söylediniz — peki neden en baştan küçültüyordunuz?**
Çünkü standart görüntü sınıflandırma tarifi bu: GPU aynı boyutta girdi ister, hazır ağlar 224 pikselle eğitilmiştir, ve 1024 piksellik görsel işlemesi 21 kat pahalıdır — bizde 4 dakika yerine 80 dakika. Ama o tarif "kedi mi köpek mi" için yazılmış; kedi küçültmeye dayanır, üretim izi dayanmaz. Kare kare inceleme bu üç pratik kısıtı da çözdü: her parça sabit 128 piksel, orijinal çözünürlükte.

**2. Kesinlik yüzde 94 dediniz ama test setiniz 500 AI'a karşı 100 gerçek — sınıflar dengesiz değil mi?**
Evet, dengesiz — ve tam bu yüzden kesinliği manşet sayı olarak kullanmıyoruz. Gerçek sınıfı küçük olunca yanlış alarm üretecek fotoğraf azdır ve kesinlik olduğundan iyi görünür. Bizim manşet ölçümüz bilerek farklı: görülmemiş gerçek kaynaklarda, sabit yanlış alarm bütçesinde AI yakalama oranı. Nitekim en kötü kaynakta yüzde 96 yanlış alarmı gösteren de aynı protokoldü — dengesizliğin gizleyebileceği sorunu bu ölçüm açığa çıkardı.

**3. Demo'da servis edilen modeller sizin mi?**
Hayır, ve bunu açıkça söylüyoruz: Community-Forensics ViT MIT lisanslı ve her zaman açık; B-Free CVPR 2025'ten, lisansı gereği ayrı bir bayrağın arkasında. Bizim katkımız karar katmanı: 12 kaynaklı en-kötü-kaynak kalibrasyonu, asimetrik karar bandı, 2048 piksel sınırı, sıkıştırma rozeti ve OR hüküm kuralı — her cevapta hangi deneyden geldiği yazıyor. Kendi ResNet'imiz de ekranda ama "araştırma sinyali — karara dahil değil" etiketiyle, kendi hata oranı üzerinde yazılı hâlde.

**4. Sistem GPT görsellerini neden kaçırıyor?**
Çünkü servis ettiğimiz iki modelin ikisi de GPT Image ailesini eğitiminde görmedi ve bu ailenin bıraktığı izler, öğrendikleri difüzyon izlerinden farklı — OR kuralıyla bile yakalama yüzde 12'de kalıyor. Bu, literatürün de doğruladığı genel bir durum: donmuş dedektörler yeni nesil üreteçlerde ciddi düşüyor. Aday çözümümüz GPT verisiyle eğitilmiş üçüncü bir kol; kabul şartını da baştan yazdık: yanlış alarm bütçesini bozmayacak. Bu arada sistemin dürüst tarafı şu: kaçırdığında "gerçektir" demiyor, "yeterli kanıt yok" diyor.

**5. Sistem bir görsele hiç "gerçek" diyor mu?**
Hayır, ve bu bilinçli bir tasarım kararı — deney numarasıyla kayıtlı. E23a'da ölçtük: "gerçektir" kararı hiçbir kaynakta tutarlı verilemiyordu, bir kaynakta sıfır kapsamdaydı — ve daha kötüsü, AI içerik o kapıdan "gerçek" damgası alıp aklanabilirdi. O yüzden karar iki seçenekli: "AI tespit edildi" ya da "yeterli kanıt yok". Yokluğu kanıtlayamayan bir sistemin bunu iddia etmemesi gerektiğini düşünüyoruz.

**6. Yarın yeni bir üreteç çıksa sisteminiz ne yapar?**
İki katmanlı cevabım var. Modeller o üreteci muhtemelen ilk başta kaçırır — bunu GPT örneğinde zaten ölçtük ve sistem bu durumda yanlış güven vermek yerine "yeterli kanıt yok" der. Ama karar katmanı üreteçten bağımsız çalışır: yeni bir kamera ya da işleme hattı için tarif belli — denetle, 2048'e sınırla, yaklaşık 100 fotoğrafla kalibre et, eşiği güncelle. Bu tarifi iki kez, en son kendi telefonumun 207 fotoğrafıyla doğruladık.

**7. Yüzde 65 yakalama düşük değil mi? Üç sahteden biri kaçıyor.**
Tek başına bakınca düşük görünüyor, ama o sayı bir takasın sonucu: yüzde 10 yanlış alarm bütçesini hiçbir kaynakta aşmama şartıyla yüzde 65. Eşiği düşürüp daha çok yakalayabiliriz — ama o zaman masum fotoğrafları suçlamaya başlarız, ki bir haber fotoğrafına "sahte" demek, bir sahteyi kaçırmaktan daha pahalı bir hata. Ayrıca kaçan görseller "gerçek" ilan edilmiyor, "yeterli kanıt yok" bandında kalıyor. Karşılaştırma için: bu bütçede eski modelimizin yakalaması yüzde 1'lere düşüyordu.

**8. İnternetten geçmiş, ağır sıkıştırılmış görsellerde sistem çalışıyor mu?**
Kısmen — ve bunu ölçtük, saklamıyoruz. E23c gösterdi ki sıkıştırma ayrı bir eşik dünyası: bir kol güvenli tarafta kalıyor, diğeri sıkıştırma refit'i olmadan tehlikeli şekilde yanılabiliyor; rejime göre eşik yenilendiğinde bütçe korunuyor ama yakalama düşüyor. Sistem bu düşüşü kullanıcıdan gizlemiyor — görselin sıkıştırma rejimini tespit edip sonucun yanında rozet olarak gösteriyor. Ağır sıkıştırma için kalıcı çözüm, eğitime sıkıştırma çeşitlemesi eklemek — açık kalanlar listemizde.
