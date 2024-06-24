
# SignLanguageDetection

Türkçe işaret dilini **gerçek zamanlı** algılayan ve Türkçe **doğal dil işleme (NLP)** tekniklerini kullanan bir yapay zeka modelidir. 
Model sayesinde işaret dilindeki harfler ve bazı temel kelimeler tanınır. Tanınan kelimelerle oluşturulan cümleler, doğal dil işleme teknikleriyle çekimlenerek gramer açısından doğru ve anlaşılır bir hale getirilir.

## Gerekli Kurulumlar

Projenin kodlanması **PyCharm**’da **Python**’un 3.10 versiyonu ile gerçekleştirilmiştir. TensorFlow 2.12.0, Keras 2.12.0, MediaPipe 0.10.9, OpenCV 4.8.1.78, NumPy 1.23.5 ve Sklearn 1.3.2 sürümlü araçları yüklenmiştir. TensorFlow ve Keras'ın uyumlu olmasına dikkat edilmelidir.

## Kodu Test Etme

Kodu test etmek için yapılması gerekenler:

1) Sing_Lang_Detection_letters.py dosyasını açın.

2) Gerekli araçları yükleyin.

3) Eğer modeli eğittikten sonra proje test edilmek istenirse 161 ve 170. satırları açın. 173. satırı kapatın. Ağırlık kaydetme aşamasında var olan ağırlığın üzerine kaydetme olmaması amacıyla 170. satırdaki ağırlık adı değiştirilmelidir.

Modeli eğitmeden proje çalıştırılmak istenirse 161 ve 170. satırları kapatın. 173. satırı açın. Ağırlıkların yüklenmesi aşamasında 173. satırdaki ağırlık isminin istenen ağırlıkla uyumlu olmasına dikkat edilmelidir.

4) Ardından kodu çalıştırarak test edin.

Çekimli fiiller kodunu test etmek için yapılması gerekenler:

1) Sing_Lang_Detection_words.py dosyasını açın.

2) Gerekli araçları yükleyin.

3) Eğer modeli eğittikten sonra proje test edilmek istenirse 191 ve 195. satırları açın. 198. satırı kapatın. Ağırlık kaydetme aşamasında var olan ağırlığın üzerine kaydetme olmaması amacıyla 195. satırdaki ağırlık adı değiştirilmelidir.

Modeli eğitmeden proje çalıştırılmak istenirse 191 ve 195. satırları kapatın. 198. satırı açın. Ağırlıkların yüklenmesi aşamasında 198. satırdaki ağırlık isminin istenen ağırlıkla uyumlu olmasına dikkat edilmelidir.

4) Ardından kodu çalıştırarak test edin.

5) NLP (doğal dil işleme) modeli yeniden eğitilmek istenirse 50 ve 51. satırları açın. 54. satırı kapatın. Ağırlık kaydetme aşamasında var olan ağırlığın üzerine kaydetme olmaması amacıyla 51. satırdaki ağırlık adı değiştirilmelidir.

Modeli eğitmeden proje çalıştırılmak istenirse 50 ve 51. satırları kapatın. 54. satırı açın. Ağırlıkların yüklenmesi aşamasında 51. satırdaki ağırlık isminin istenen ağırlıkla uyumlu olmasına dikkat edilmelidir.

6) Yeni NLP modeliyle kodu çalıştırarak test edin.

## Test Aşaması Çıktıları
Kameradan alınan insan eli görüntüsü üzerinde **landmarklar** oluşturularak hareket tanınır. Demo için giriş verileri olarak Türkçe alfabesindeki 29 harf ve bazı temel kelimeler seçilmiştir. 

Modelin gerçek zamanlı olarak video akışındaki harfleri ve kelimeleri tanıdığı, kelimelerle oluşturulan cümlelerin de **çekimlendiği** test aşamasında yapılan tahminler sonucunda maksimum **tahmin olasılığına** sahip harf, kelime veya çekimlenmiş cümle ekrana yazdırılmıştır. Örnek çıktılar aşağıdaki gibidir.

### F HARFİ:
![F harfi](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/dfef2e58-576f-40a5-bb2c-9a1c08d9dfbc)

### M HARFİ:
![M harfi](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/259f9936-45c9-4c7c-82c8-88e9bb4a189e)

### Z HARFİ:
![Z harfi](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/a2859363-0b68-4650-b4b5-316f06972ff7)

### 'BUGÜN GELİYORUM' CÜMLESİ:
![bugun geliyorum](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/1684b445-e2c7-4fe9-9c48-3cab08d6210e)

### 'BUGÜN GELİYORSUN' CÜMLESİ:
![bugun geliyorsun](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/5b971752-4913-47db-8d7f-9e0271e430ce)

### 'YARIN GELECEĞİM' CÜMLESİ:
![yarin gelecegim](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/80e8da17-2420-4574-b350-78efb81631d6)

### 'YARIN GELECEKSİN' CÜMLESİ:
![yarin geleceksin](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/b437c991-825d-41ec-a4be-401bf0117efe)

## Ekip Arkadaşlarım

* **Aysun KOYLU** - [cengak](https://github.com/cengak)

* **Öykü ÜNSAY** - [oykunsay](https://github.com/oykunsay)

* **Elif Nur ÖZDEMİR** - [elifnurozdemir](https://github.com/elifnurozdemir)

## License
Bu proje GPL-3.0 Lisansı altında lisanslanmıştır. Ancak bu pojede yer alan veriler yazar tarafından toplanmıştır ve ticari amaçlarla kullanılmasına izin verilmez. Kullanıcıların kodu ticari uygulamalar için kullanmak üzere kendi verilerini toplamaları gerekir.

## Bilgilendirme
Bu projedeki veriler ben ve ekip arkadaşlarım tarafından toplanmış ve eğitilmiştir. Daha iyi bir yapay zeka modeli için geniş ve çeşitli işaret dilini içeren bir veri seti geliştirilebilir. Bu veri seti çeşitli işaret dili varyasyonlarını ve hareketlerini kapsamalıdır.
Proje geliştirilirken [bir YouTube videosundan](https://youtu.be/doDUihpj6ro?si=28wMFGQ5kvq6czgK) ilham alınmıştır.