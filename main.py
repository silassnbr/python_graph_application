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
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from tkinter import Label, Tk, Button, filedialog,messagebox
import re
import string
# dosya seçme fonksiyonu 
ozel_isimler = []
sayilar=[]
cumle_uz=[]
skor_ozel=[]
skor_numerik=[]
duzenlenmisCumleler=[]
def dosya_bul():
    root = Tk()
    root.withdraw()
    dosya_yolu = filedialog.askopenfilename()
    if dosya_yolu:
        ozel_isimler.clear()
        sayilar.clear()
        cumle_uz.clear()
        skor_ozel.clear()
        skor_numerik.clear()
        duzenlenmisCumleler.clear()
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
        #Metin noktaya göre ayırılıp diziye atanır.
        G = nx.Graph()
        cumleler = dosya_icerigi.split(".")
    # graph olustruma kısmı DÜZENLENECEK###############################
    for i in range(len(cumleler)-1):
        G.add_node(cumleler[i],label=cumleler[i])

    for i in range(len(cumleler) - 2):
        G.add_edge(cumleler[i],cumleler[i+1])
    #########################
    for i in range(len(cumleler)-1):
        ozel_isimSay(cumleler[i])
        numerikSayisi(cumleler[i])
        cumleUzunlugu(cumleler[i])
        skorDonustur(ozel_isimler[i],cumle_uz[i],sayilar[i])
    for i in range(len(cumleler)-1):
        nltkAsdimlari(cumleler[i])
    label_sayiOzel.config(text=f"Özel İsim skor: {skor_ozel}")
    label_sayi.config(text=f"Numerik skor: {skor_numerik}")
    labelCumleUz.config(text=f"{duzenlenmisCumleler}")
    for a in range(len(duzenlenmisCumleler)-1):
        print(duzenlenmisCumleler[a])
    nx.draw(G, with_labels=True)
    plt.show()

def nltkAsdimlari(duzenle):
    stemmer = PorterStemmer() 
    
    punctuation = string.punctuation
    # Cümleyi punctuation içindeki karakterlere göre bölerek kelimelere ayırın
    # tokensPunc = duzenle.split()
    # Noktalama işaretleri olmayan kelimeleri filtreleyin
    punct_sentence = ''.join(char for char in duzenle if char not in punctuation)
    # Filtrelenmiş kelimeleri birleştirerek cümleyi oluşturun
    # punct_sentence = ' '.join(punct_tokens)
    
    
    stop_words = set(stopwords.words("english"))  # İngilizce stop words'leri yükleyin
    stop_tokens = word_tokenize(punct_sentence)  # Cümleyi kelimelere ayırın
    filtered_tokens = [token for token in stop_tokens if token.lower() not in stop_words]  # Stop words olmayan kelimeleri filtreleyin
    filtered_sentence = ' '.join(filtered_tokens) 


    duzenlenmis = sent_tokenize(filtered_sentence)
    stemmed_tokens = [stemmer.stem(token) for token in filtered_sentence]  # Her kelimeye stemming işlemi uygulayın
    stemmed_sentence = ' '.join(stemmed_tokens)

    

    
    duzenlenmisCumleler.append(filtered_sentence)
def skorDonustur(ozel,cumle,numer):
    a=round(float(ozel/cumle),3)
    skor_ozel.append(a)
    b=round(float(numer/cumle),3)
    skor_numerik.append(b)
def cumleUzunlugu(cumle):
    s=cumle.split()
    cumle_uz.append(len(s))
# dosyanın txt olup omadığını kontrol eden fonksiyon
def txt_kontrol(dosya_yolu):
    dosya_uzantısı=os.path.splitext(dosya_yolu)[1]
    if dosya_uzantısı.lower()==".txt":
        return True
    else:
        return False
#özel isim sayısı bulma
def ozel_isimSay(metin):
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

labelCumleUz = Label(root, text="",fg="#643843")
labelCumleUz.pack(pady=10)
labelCumleUz.configure(bg="#C88EA7")
root.mainloop()


