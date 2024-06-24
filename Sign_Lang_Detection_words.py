import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM

mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils  # drawing utilities

# Örnek veri
cümleler = [
    "ben yarin gelmek", "ben bugun gelmek", "sen yarin gelmek", "sen bugun gelmek"
]
cekimler = [
    "ben yarin gelecegim", "ben bugun geliyorum", "sen yarin geleceksin", "sen bugun geliyorsun"
]

# Tokenizer'ı eğitmek
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(cümleler + cekimler)

# Tokenizer'ı kullanarak dizileri sayılara çevirmek
X = tokenizer.texts_to_sequences(cümleler)
y = tokenizer.texts_to_sequences(cekimler)

# Dizileri aynı uzunlukta yapmak
max_len = 16
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')

# Çıkış verisini reshaped yapıyoruz
y = np.expand_dims(y, -1)

# Model oluşturma
modelt = Sequential()
modelt.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=max_len))
modelt.add(Bidirectional(LSTM(50, return_sequences=True)))
modelt.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Modeli derleme
modelt.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme ve ağırlıkları kaydetme
# model.fit(X, y, epochs=400, batch_size=1)
# model.save('fiil_cekim_modeli.h5')

# Modeli yükleme
modelt = load_model('fiil_cekim_modeli.h5')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    image.flags.writeable = False  # non-writeable image
    results = model.process(image)  # make prediction
    image.flags.writeable = True  # writeable image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB -> BGR
    return image, results


def draw_styled_landmarks(image, results):
    # eklenen iki değişkenden ilki dot style, ikincisi line style için                        )
    # Sol el bağlantıları
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

    # Sağ el bağlantıları
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    # landmarks mevcut mu? mevcutsa bir array oluştur, değilse sıfırlardan oluşan bir array oluştur.
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:

        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([lh, rh])  # dizi uzunluğu = 21*3+21*3 = 126 keypoint


# COLLECTIONLAR İÇİN KLASÖR OLUŞTURMA
# verilerin, numpy arrays olarak tutulacağı adres
DATA_PATH = os.path.join('MP_Data/kelimeler')
actions = np.array(['merhaba', 'ben', 'sen', 'iyi', 'yarin', 'bugun', 'gelmek', 'tesekkurler', 'nasil', 'hoscakal'])

num_sequences = 30  # 30 video
sequence_length = 30  # her video için 30 kare
# her karede 126 landmark değeri

for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# # VERİLERİN TOPLANMASI
# cap = cv2.VideoCapture(0)
# # mediapipe modelini setleme
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#
#     # her kare döngüye girer
#     for action in actions:
#         for sequence in range(num_sequences):   # video no
#             for frame_num in range(sequence_length):   # kare no
#
#                 # kameradan kare kare okuma
#                 ret, frame = cap.read()
#
#                 # sonuçlar results değişkeninde tutulacak (tespit aşaması)
#                 image, results = mediapipe_detection(frame, holistic)
#
#                 # landmarks
#                 draw_styled_landmarks(image, results)
#
#                 # NEW Apply wait logic
#                 if frame_num == 0:
#                     cv2.putText(image, 'YENI VERI ICIN HAZIR OLUN', (120, 200),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, '{} icin kareler toplaniyor.. Video No: {}'.format(action, sequence), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # ekranda gösterme
#                     cv2.imshow('TSLR', image)
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(image, '{} icin kareler toplaniyor.. Video No: {}'.format(action, sequence), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # ekranda gösterme
#                     cv2.imshow('TSLR', image)
#
#                 keypoints = extract_keypoints(results)     # 126 keypoint
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)
#
#                 # q ile çıkış
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break
#
#     cap.release()
#     cv2.destroyAllWindows()

# ETİKETLERİ OLUŞTURMA
# Label map ve kategorik temsil
label_map = {label: num for num, label in enumerate(actions)}  # {'a': 0, 'b': 1,..., 'd': 4}
sequences, labels = [], []
for action in actions:
    for sequence in range(num_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
# print(np.array(sequences).shape)  # (150, 30, 126)

# Test ve eğitim verilerinin ayrılması
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# print(X_train.shape)   # sequences (142, 30, 126)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  # actions.shape[0] = 5
# softmax -> 0 ile 1 arasında bir olasılık değeri

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=250, callbacks=[tb_callback])  # tensor board
# model.summary()

# # AĞIRLIKLARI KAYDETME
# model.save('action2.h5')

# AĞIRLIKLARI YÜKLEME
model.load_weights('action2.h5')

# CONFUSION MATRIX
# TAHMİN ETME
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()  # [1, 1, 0, 0, 0]
# print(multilabel_confusion_matrix(ytrue, yhat))
# print(accuracy_score(ytrue, yhat))


# TEST AŞAMASI
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 70 + num * 40), (int(prob * 100), 100 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 100 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# Fonksiyon: Verilen fiili istenilen formata göre çekimleme
def cekimleme(model, tokenizer, cumle):
    seq = tokenizer.texts_to_sequences([cumle])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')
    tahmin = model.predict(seq)
    tahmin_kelime = [tokenizer.index_word[np.argmax(t)] for t in tahmin[0]]
    return ''.join(tahmin_kelime).strip()


sequence = []
sentence = []
cekimlenmis_sentence = ''
allowed_words = ['ben', 'sen', 'yarin', 'bugun', 'gelmek']
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# mediapipe modelini setleme
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # kameradan kare kare okuma
        ret, frame = cap.read()

        # sonuçlar results değişkeninde tutulacak (tespit aşaması)
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # landmarks
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)  # 126 keypoint
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Tahmin etme
        # x_test iki argümana sahip (30,126)
        # (num_sequences, 30, 126) olması bekleniyor
        # bunun için yazılan: np.expand_dims(sequence, axis=0)
        # 0.'ya sequence eklenir
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # 3. Visulation logic
            # result, threshold'tan büyük mü?
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 7:
                sentence = sentence[-7:]

            # Sentence çekimleme işlemi
            if len(sentence) > 0:
                raw_sentence = [word for word in sentence if word in allowed_words]

                # "bugun" ve "yarin" kelimelerinin aynı anda bulunmamasını sağla
                if 'bugun' in raw_sentence and 'yarin' in raw_sentence:
                    # Eğer her ikisi de varsa, "yarin" çıkarılıyor
                    raw_sentence = [word for word in raw_sentence if word != 'yarin']

                if len(raw_sentence) > 3:
                    raw_sentence = raw_sentence[-3:]

                    # raw_sentence uzunluğunu 3 kelimeye tamamlama
                if len(raw_sentence) < 3:
                    for word in reversed(sentence):
                        if word in allowed_words and word not in raw_sentence:
                            raw_sentence.append(word)
                        if len(raw_sentence) == 3:
                            break

                raw_sentence_str = ' '.join(raw_sentence)
                cekimlenmis_sentence = cekimleme(modelt, tokenizer, raw_sentence_str)


            # Visulation probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (255, 175, 97), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (0, 40), (640, 75), (255, 191, 0), -1)
        cv2.putText(image, cekimlenmis_sentence, (3, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Ekranı tam ekran yapma
        cv2.namedWindow('OpenCV TSLR', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('OpenCV TSLR', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Ekranda gösterme
        cv2.imshow('OpenCV TSLR', image)

        # q ile çıkış
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()