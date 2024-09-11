import cv2
import mediapipe as mp
import threading
import pyttsx3

def count_fingers(points):
    dedos = [8, 12, 16, 20]
    contador = 0

    if points:
        # Verificar se o polegar está levantado
        if points[4][0] < points[3][0]:
            contador += 1
        for x in dedos:
            if points[x][1] < points[x-2][1]:
                contador += 1

    return contador

def speak_count(count):
    engine = pyttsx3.init()
    engine.say(f"{count} dedos levantados")
    engine.runAndWait()

def process_frame(video, hand, mpDraw):
    previous_count = -1  # Valor inicial que não pode ser um número de dedos válido

    while True:
        check, img = video.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hand.process(imgRGB)
        handsPoints = results.multi_hand_landmarks
        h, w, _ = img.shape
        pontos = []
        if handsPoints:
            for points in handsPoints:
                mpDraw.draw_landmarks(img, points, mp.solutions.hands.HAND_CONNECTIONS)
                for id, cord in enumerate(points.landmark):
                    cx, cy = int(cord.x * w), int(cord.y * h)
                    pontos.append((cx, cy))

            contador = count_fingers(pontos)

            if contador != previous_count:
                previous_count = contador
                # Falar o número de dedos levantados em uma thread separada
                threading.Thread(target=speak_count, args=(contador,)).start()

            cv2.rectangle(img, (80, 10), (200, 100), (255, 0, 0), -1)
            cv2.putText(img, str(contador), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    # Inicializar a detecção de mãos
    video = cv2.VideoCapture(0)  # Ajustar o índice da câmera se necessário
    hand = mp.solutions.hands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    try:
        process_thread = threading.Thread(target=process_frame, args=(video, hand, mpDraw))
        process_thread.start()
        process_thread.join()
    except KeyboardInterrupt:
        print("Interrupção pelo usuário")

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
