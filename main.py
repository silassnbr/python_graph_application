import random
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import os
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from tkinter import Label, Tk, Button, filedialog,messagebox
# dosya seçme fonksiyonu 
def dosya_bul():
    root = Tk()
    root.withdraw()
    dosya_yolu = filedialog.askopenfilename()
    if dosya_yolu:
        if txt_kontrol(dosya_yolu):
            node_olustur(dosya_yolu)
        else:
            messagebox.showinfo("UYARI","Lütfen txt formatında bir dosya seçiniz")
    else:
        messagebox.showinfo("UYARI","Dosya Seçilemedi")
#node oluşturma fonksiyonu
def node_olustur(dosya_yolu):
    with open(dosya_yolu, 'r') as f:
        dosya_icerigi = f.read()
        G = nx.Graph()
        cumleler = dosya_icerigi.split(".")

    for i in range(len(cumleler)):
        G.add_node(cumleler[i],label=cumleler[i])

    for i in range(len(cumleler) - 1):
        G.add_edge(cumleler[i],cumleler[i+1])


    nx.draw(G, with_labels=True)
    plt.show()

# dosyanın txt olup omadığını kontrol eden fonksiyon
def txt_kontrol(dosya_yolu):
    dosya_uzantısı=os.path.splitext(dosya_yolu)[1]
    if dosya_uzantısı.lower()==".txt":
        return True
    else:
        return False

root = Tk()
root.title("Dosya Seçme Uygulaması")
root.configure(bg="#C88EA7")
yaziboyutu=("Arial",16)
etiket = Label(root, text="LÜTFEN DOSYA SEÇİNİZ",font=yaziboyutu,fg="#643843",anchor='w')
etiket.pack(pady=20)
etiket.configure(bg="#C88EA7")

pencere_genislik = 800
pencere_yukseklik = 600
ekran_genislik = root.winfo_screenwidth()
ekran_yukseklik = root.winfo_screenheight()
x_konumu = int((ekran_genislik - pencere_genislik) / 2)
y_konumu = int((ekran_yukseklik - pencere_yukseklik) / 2)
root.geometry(f"{pencere_genislik}x{pencere_yukseklik}+{x_konumu}+{y_konumu}")

buton = Button(root, text="DOSYA SEÇ", command=dosya_bul,border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
buton.pack(pady=10)
buton.configure(bg="#99627A")

root.mainloop()


