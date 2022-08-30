import cv2

for i in range(65535):
    cap = cv2.VideoCapture(6)
    ret, frame = cap.read()
    if ret is None:
        continue
    edframe = frame
    cv2.putText(edframe, 'cap:{}'.format(i), (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)

    # 加工済の画像を表示する
    cv2.imshow('Edited Frame', edframe)

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(0)
    cap.release()
    if k == 27:
        break

cv2.destroyAllWindows()
