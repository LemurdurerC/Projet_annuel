from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import PIL.Image



class App():
    def __init__(self):
        self.file=None
        def showInfo(msg):
            messagebox.showinfo("Erreur",msg)

        def resize(img):
            new_image = img.resize((256,256))
            return new_image

        def upload():
            self.file = filedialog.askopenfile(parent=fenetre, mode='rb', title='Choose a file')
            if self.file is None:
                showInfo("Veuillez choisir une image valide")
                return
            filestr = self.file.name
            filestr = filestr.split(".")[1]
            if self.file and filestr.__contains__("png"):
                print(self.file.name.split(".")[0])
                #image = PhotoImage(file=self.file.name)
                image = PIL.Image.open(self.file.name)
                image = resize(image)
                image = image.convert("RGB")
                image.save(self.file.name.split(".")[0]+"_bis.png", "PNG")
                image = PhotoImage(file=self.file.name.split(".")[0]+"_bis.png")
                label.config(image=image)
                label.photo_ref=image
                predict_appear()
            else:
                showInfo("Veuillez choisir une image valide")

        def predict_appear():
            predict_button.place(x="305", y="440")


        def predict():
            filestr = self.file.name
            filestr = filestr.split(".")[1]
            if filestr.__contains__("jpg") or filestr.__contains__("png"):
                print('dosomething')
            else:
                showInfo("Veuillez choisir un format valide")


        fenetre = Tk()
        fenetre.geometry("800x600")
        fenetre.title("Test")
        #fenetre['bg'] = "#E2474B"
        canvas = Canvas(fenetre, width=800, height = 600)
        cedric = PhotoImage(file="IMAGES/Fond.png")
        canvas.create_image(0,0,anchor=NW,image=cedric)
        canvas.pack()
        fenetre.resizable(height=False, width=False)

        width = 300
        height = 300
        image = PhotoImage(file="testi.png")

        v = IntVar()
        button_test = PhotoImage(file="refa.png")
        #linear_img = PhotoImage(file ="LINEAR.png")
        linear_img = PhotoImage(file ="IMAGES/LINEAR2.png")
        #MLP_img = PhotoImage(file = "MLP.png")
        MLP_img = PhotoImage(file = "IMAGES/MLP.png")
        #SVM_img = PhotoImage(file = "SVM.png")
        SVM_img = PhotoImage(file = "IMAGES/SVM.png")
        #RBF_img = PhotoImage(file = "RBF.png")
        RBF_img = PhotoImage(file="IMAGES/RBF.png")
        #img_titre = PhotoImage(file = "mood_64.png")
        #txt_titre = PhotoImage(file = "Titre.png")
        txt_titre = PhotoImage(file="IMAGES/mood.png")
        #predict_img = PhotoImage(file="start-button.png")
        predict_img = PhotoImage(file="IMAGES/POWER.png")

        s1 = Radiobutton(fenetre, text="One", variable=v, value=1,image=linear_img,bg="black", activebackground ="black")
        s1['border']="0"
        s1.place(anchor=W,x="0", y="200")
        Radiobutton(fenetre, text="Two", variable=v, value=2,image=MLP_img,bg="#E2474B", activebackground ="#E2474B").place(anchor=W,x="0", y="250")
        Radiobutton(fenetre, text="One", variable=v, value=3,image=SVM_img,bg="#E2474B", activebackground ="#E2474B").place(anchor=W,x="0", y="300")
        Radiobutton(fenetre, text="Two", variable=v, value=4,image=RBF_img,bg="#E2474B", activebackground ="#E2474B").place(anchor=W,x="0", y="350")
        """image=img_titre,"""
        titre_img = Label(fenetre,  bg="#E2474B",borderwidth=0)
        titre_img.place(x="650", y="0")

        titre_txt = Label(fenetre, image=txt_titre, bg="#E2474B",borderwidth=0)
        titre_txt.place(x="250", y="25")

        label = Label(fenetre, image=image, bg="#E2474B",borderwidth=0)
        label.place(x="250", y="160")

        predict_button = Button(fenetre, borderwidth=0, image=predict_img, bg="#E2474B", activebackground ="#E2474B", command=lambda: predict())
        predict_button.pack_forget()

        browse = Button(fenetre, command=lambda: upload(), borderwidth=0, image=button_test, bg="#E2474B", activebackground ="#E2474B")
        browse.place(x="560", y="250")




        fenetre.mainloop()
app = App()