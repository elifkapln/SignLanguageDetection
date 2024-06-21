import cv2
import numpy as np
import os
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils  # drawing utilities


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
    #landmarks mevcut mu? mevcutsa bir array oluştur, değilse sıfırlardan oluşan bir array oluştur.

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
DATA_PATH = os.path.join('MP_Data/harfler')
actions = np.array(['a', 'b', 'c', 'cc', 'd', 'e', 'f', 'g', 'gg', 'h', 'ii', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'oo', 'p', 'r', 's', 'ss', 't', 'u', 'uu', 'v', 'y', 'z'])
# 30 video
num_sequences = 30
# her video için 30 kare
sequence_length = 30
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
label_map = {label: num for num, label in enumerate(actions)}   # {'a': 0, 'b': 1,..., 'd': 4}
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

# res = [.7, 0.2, 0.1]
# print(actions[np.argmax(res)])  # a

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=250, callbacks=[tb_callback])  # tensor board
# model.summary()

# # TAHMİN ETME
# res = model.predict(X_test)
# print(actions[np.argmax(res[4])])
# print(actions[np.argmax(y_test[4])])

# # AĞIRLIKLARI KAYDETME
# model.save('action.h5')

# AĞIRLIKLARI YÜKLEME
model.load_weights('action.h5')

# CONFUSION MATRIX
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()      # [1, 1, 0, 0, 0]
# print(multilabel_confusion_matrix(ytrue, yhat))
# print(accuracy_score(ytrue, yhat))


# TEST AŞAMASI
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    frame_width = output_frame.shape[1]
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 85 + num * 40), colors[num], -1)
        action_text = actions[num][:15]
        cv2.putText(output_frame, action_text, (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    for num, prob in enumerate(res):
        rectangle_width = int(prob * 100)
        rectangle_x = frame_width - rectangle_width
        cv2.rectangle(output_frame, (rectangle_x, 60 + num * 40), (frame_width, 85 + num * 40), colors[num], -1)
        action_text = actions[num]  # Tüm metni al
        text_x = frame_width - rectangle_width - 10
        text_y = 80 + num * 40  # Dikdörtgenin y koordinatı
        cv2.putText(output_frame, action_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return output_frame



sequence = []
sentence = []
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

        keypoints = extract_keypoints(results)     # 126 keypoint
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

            if len(sentence) > 10:
                sentence = sentence[-10:]

            # Visulation probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Ekranı tam ekran yapmaz
        cv2.namedWindow('OpenCV TSLR', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('OpenCV TSLR', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Ekranda gösterme
        cv2.imshow('OpenCV TSLR', image)

        # q ile çıkış
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()