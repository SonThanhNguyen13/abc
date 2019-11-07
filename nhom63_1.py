import os
from random import shuffle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageTk
import tkinter as tk


# make a class Application
class Application:
    def __init__(self):
        print("[INFO]: starting...")
        # turn of webcam when class is called
        self.vs = cv2.VideoCapture(0)# capture video frames
        self.current_image = None  # current image from the camera
        self.gray_img = None # current gray image
        self.thre_img = None # current binary image
        self.model = train_model('english-alphabets') # train model for predict
        self.root = tk.Tk()  # initialize root window
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.creat_widgets() # create wideget for windows
        self.video_loop() # play video

    def creat_widgets(self):
        # Window
        self.root.title("Demo") # set title
        self.root.resizable(height=False, width=False) # make window unresizable
        self.root.minsize(height=500, width=1000) # size for window
        # FRAME
        self.frame1 = tk.Frame(self.root, width=600, height=1000)
        self.frame1.place(x=10, y=10)
        self.frame2 = tk.Frame(self.root, width=500, height=300)
        self.frame2.place(x=620, y=10)
        self.frame3 = tk.Frame(self.root, width = 500, height = 300)
        self.frame3.place(x=620, y=250)
        # PICTURES
        self.lmain = tk.Label(self.frame1)
        self.lmain.place(x=0, y=0)
        self.im_gray = tk.Label(self.frame2)
        self.im_gray.place(x = 0, y = 0)
        self.im_thre = tk.Label(self.frame3)
        self.im_thre.place(x=0, y=0)

    def destructor(self):
        print("[INFO] closing...")
        self.vs.release()
        self.root.destroy() # destroy window
        cv2.destroyAllWindows()

    def video_loop(self):
        ok, frame = self.vs.read()  # read frame from video stream
        if ok:  # frame captured without any errors
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame into gray
            #gray = change_brightness(gray, 1.6, 10)
            small = cv2.resize(gray, (350, 240)) # make a small img to display
            _, thre = cv2.threshold(src=small, thresh=160, maxval=255, type=cv2.THRESH_BINARY_INV) # make binary from small image
            _, pr_thre = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV) # make binary frame to process
            # find contours in binary
            contours, _ = cv2.findContours(pr_thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in contours:
                (x, y, w, h) = cv2.boundingRect(i) # make a rectangle around contours
                if (w > 15 and h > 15) and (w < 100 and h < 100):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1) # draw rectangles
                    roi = pr_thre[y:y + h, x:x + w] # cut letters
                    roi= cv2.equalizeHist(roi)
                    roi = cv2.resize(roi, (34, 34)) # resize to match train data
                    roi = np.array(roi)
                    roi = roi.reshape(-1, (34 * 34)) # reshape to match train data
                    _, results, _, _ = self.model.findNearest(np.float32(roi), 4) # predict
                    pre = chr(results)
                    cv2.putText(frame, pre, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1) # put text
            # Show images
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert from RGB to RBGA for Imagetk
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            self.gray_img = Image.fromarray(small)
            self.thre_img = Image.fromarray(thre)
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            graytk = ImageTk.PhotoImage(image=self.gray_img)
            thretk = ImageTk.PhotoImage(image=self.thre_img)
            self.lmain.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.lmain.config(image=imgtk)  # show the image
            self.im_gray.imgtk = graytk
            self.im_gray.config(image=graytk)
            self.im_thre.imgtk = thretk
            self.im_thre.config(image=thretk)
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds


def create_data(link):
    print('[INFO]: creating data...')
    path = link
    folders = []
    img_paths = []
    data = []
    # get all folders in path
    for i in os.listdir(path):
        folders.append(os.path.join(path, i))
    for folder in folders:
        for img_path in os.listdir(folder):
            img_paths.append(os.path.join(folder, img_path))
    shuffle(img_paths)
    labels = [i[18] for i in img_paths]
    labels = [ord(i) for i in labels]
    for i in img_paths:
        img = cv2.imread(i) # read images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert into gray images
        img = cv2.equalizeHist(img)
        im,thre = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        np_img = np.array(thre)
        data.append(np_img)
    data = np.array(data)
    print('[INFO]: done...')
    return data, labels


def train_model(link):
    data, labels = create_data(link)
    print('[INFO]: creating model...')
    data_shape = data.shape # get datashape to create train data
    data = data.reshape(data_shape[0], data_shape[1] * data_shape[2]) #reshape
    # split data into train set and test set
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=20)
    # convert into match train model data type
    train_data = np.float32(train_data)
    train_labels = np.float32(train_labels)
    test_data = np.float32(test_data)
    test_labels = np.float32(test_labels)
    # create model & train
    model = cv2.ml.KNearest_create()
    model.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    print('[INFO]: testing model...')
    # predict
    _, results, _, _ = model.findNearest(test_data, 4)
    # test accuracy on test data
    accuracy = accuracy_score(results,  test_labels)
    print('[INFO]:  done...')
    print('Accuracy of model: {}'.format(accuracy * 100))
    return model

def change_brightness(im, alpha, beta):
    new_image = np.zeros(im.shape, im.dtype)
    for x in range(im.shape[0]):
        new_image[x] = np.clip(alpha * im[x] + beta, 0, 255)
    return new_image


def main():
    app = Application()
    app.root.mainloop()


if __name__ == '__main__':
    main()
