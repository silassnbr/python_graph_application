import random
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from tkinter import Label, Tk, Button, filedialog

def dosya_bul():
    root = Tk()
    root.withdraw()
    dosya_yolu = filedialog.askopenfilename()
    if dosya_yolu:
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

    else:
        print("Dosya seçilmedi.")

# Ana pencereyi oluştur
root = Tk()
root.title("Dosya Seçme Uygulaması")
root.configure(bg="#C88EA7")
yaziboyutu=("Arial",16)
etiket = Label(root, text="LÜTFEN DOSYA SEÇİNİZ",font=yaziboyutu,fg="#643843",anchor='w')
etiket.pack(pady=20)
etiket.configure(bg="#C88EA7")
# Pencere boyutunu ayarla
pencere_genislik = 800
pencere_yukseklik = 600
ekran_genislik = root.winfo_screenwidth()
ekran_yukseklik = root.winfo_screenheight()
x_konumu = int((ekran_genislik - pencere_genislik) / 2)
y_konumu = int((ekran_yukseklik - pencere_yukseklik) / 2)
root.geometry(f"{pencere_genislik}x{pencere_yukseklik}+{x_konumu}+{y_konumu}")
# Dosya seçme düğmesini oluştur
buton = Button(root, text="DOSYA SEÇ", command=dosya_bul,border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
buton.pack(pady=10)
buton.configure(bg="#99627A")

# Uygulamayı başlat
root.mainloop()


