import random
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import os
from nltk.chunk import ne_chunk
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tkinter import Label, Tk, Button, filedialog,messagebox
import re
# dosya seçme fonksiyonu 
ozel_isimler = []
sayilar=[]
def dosya_bul():
    
    root = Tk()
    root.withdraw()
    dosya_yolu = filedialog.askopenfilename()
    if dosya_yolu:
        ozel_isimler.clear()
        sayilar.clear()
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

    for i in range(len(cumleler)-1):
        G.add_node(cumleler[i],label=cumleler[i])

    for i in range(len(cumleler) - 2):
        G.add_edge(cumleler[i],cumleler[i+1])
    for i in range(len(cumleler)-1):
        ozel_isim_skor(cumleler[i])
        numerikSayisi(cumleler[i])
    label_sayiOzel.config(text=f"Özel İsim Sayısı: {ozel_isimler}")
    label_sayi.config(text=f"Numerik: {sayilar}")
    nx.draw(G, with_labels=True)
    plt.show()

# dosyanın txt olup omadığını kontrol eden fonksiyon
def txt_kontrol(dosya_yolu):
    dosya_uzantısı=os.path.splitext(dosya_yolu)[1]
    if dosya_uzantısı.lower()==".txt":
        return True
    else:
        return False
#özel isim sayısı bulma
def ozel_isim_skor(metin):
    
    kelimeler = word_tokenize(metin)
    
    etiketler = pos_tag(kelimeler)
    chunklar = ne_chunk(etiketler)
    sayi=0
    
    for etiket in etiketler:
        kelime, pos_etiketi = etiket
        if pos_etiketi == 'NNP': 
            sayi+=1 
    ozel_isimler.append(sayi)

    return ozel_isimler
def numerikSayisi(cumle):
    numerikler = re.findall(r'\d+', cumle)
    sayilar.append(len(numerikler))
    return sayilar
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

label_sayiOzel = Label(root, text="",fg="#643843")
label_sayiOzel.pack(pady=10)
label_sayiOzel.configure(bg="#C88EA7")

label_sayi = Label(root, text="",fg="#643843")
label_sayi.pack(pady=10)
label_sayi.configure(bg="#C88EA7")
root.mainloop()


