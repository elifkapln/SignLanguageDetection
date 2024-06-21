
# SignLanguageDetection

Türkçe işaret dilini **gerçek zamanlı** algılayarak metne dönüştüren bir yapay zeka modelidir. 

## Gerekli Kurulumlar

Projenin kodlanması **PyCharm**’da **Python**’un 3.10 versiyonu ile gerçekleştirilmiştir. TensorFlow 2.12.0, Keras 2.12.0, MediaPipe 0.10.9, OpenCV 4.8.1.78, NumPy 1.23.5 ve Sklearn 1.3.2 sürümlü araçları yüklenmiştir. TensorFlow ve Keras'ın uyumlu olmasına dikkat edilmelidir.

## Kodu Test Etme

Kodu test etmek için yapılması gerekenler:

1) Sing_Lang_Detection.py dosyasını açın.

2) Gerekli araçları yükleyin.

3) Eğer modeli eğittikten sonra proje test edilmek istenirse 161 ve 170. satırları açın. 173. satırı kapatın. Ağırlık kaydetme aşamasında var olan ağırlığın üzerine kaydetme olmaması amacıyla 170. satırdaki ağırlık adı değiştirilmelidir.

Modeli eğitmeden proje çalıştırılmak istenirse 161 ve 170. satırları kapatın. 173. satırı açın. Ağırlıkların yüklenmesi aşamasında 173. satırdaki ağırlık isminin istenen ağırlıkla uyumlu olmasına dikkat edilmelidir.

4) Ardından kodu çalıştırarak test edin.

### Test Aşaması Çıktıları
Kameradan alınan insan eli görüntüsü üzerinde **landmarklar** oluşturularak hareket tanınır. Demo için giriş verileri olarak Türkçe alfabesindeki 29 harf seçilmiştir. 

Bir sonraki adımda modelin gerçek zamanlı olarak video akışındaki harfleri ve kelimeleri tanıdığı, kelimelerle oluşturulan cümlelerin de çekimlendiği test aşaması yer almaktadır. Yapılan tahminler sonucunda maksimum **tahmin olasılığına** sahip harf ekrana yazdırılmıştır. Örnek çıktılar aşağıdaki gibidir.

F harfi:
![F harfi](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/dfef2e58-576f-40a5-bb2c-9a1c08d9dfbc)

M harfi:
![M harfi](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/259f9936-45c9-4c7c-82c8-88e9bb4a189e)

Z harfi:
![Z harfi](https://github.com/elifkapln/SignLanguageDetection/assets/103317445/a2859363-0b68-4650-b4b5-316f06972ff7)

## Ekip Arkadaşlarım

* **Aysun KOYLU** - [cengak](https://github.com/cengak)

* **Öykü ÜNSAY** - [oykunsay](https://github.com/oykunsay)

* **Elif Nur ÖZDEMİR** - [elifnurozdemir](https://github.com/elifnurozdemir)

## Bilgilendirme
Bu projedeki veriler ben ve ekip arkadaşlarım tarafından toplanmış ve eğitilmiştir. Daha iyi bir yapay zeka modeli için geniş ve çeşitli işaret dilini içeren bir veri seti geliştirilebilir. Bu veri seti, çeşitli işaret dili varyasyonlarını ve hareketlerini kapsamalıdır.