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
import torch
from transformers import BertTokenizer, BertModel
# dosya seçme fonksiyonu 
ozel_isimler = []
sayilar=[]
cumle_uz=[]
kelimesay=[]
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
        kelimesay.clear()
        if txt_kontrol(dosya_yolu):
            node_olustur(dosya_yolu)
        else:
            messagebox.showinfo("UYARI","Lütfen txt formatında bir dosya seçiniz")
    else:
        messagebox.showinfo("UYARI","Dosya Seçilemedi")
#node oluşturma fonksiyonu
def node_olustur(dosya_yolu):
    with open(dosya_yolu, encoding="utf8") as f:
        dosya_icerigi = f.read()
        satirlar=dosya_icerigi.split('\n')
        baslik=satirlar[0].strip()
        metin = '\n'.join(satirlar[1:]).strip()
        print(baslik)
        print("11111111111111111111111111111111")
        #Metin noktaya göre ayırılıp diziye atanır.
        G = nx.Graph()
        cumleler = metin.split(".")
    # graph olustruma kısmı DÜZENLENECEK###############################
    for i in range(len(cumleler)-1):
        G.add_node(cumleler[i],label=cumleler[i])

    for i in range(len(cumleler) - 2):
        G.add_edge(cumleler[i],cumleler[i+1])
    #########################
    basliktakiKelimeler=baslik.lower().split()
    for i in range(len(cumleler)-1):
        ozel_isimSay(cumleler[i])
        numerikSayisi(cumleler[i])
        cumleUzunlugu(cumleler[i])
        skorDonustur(ozel_isimler[i],cumle_uz[i],sayilar[i])

    for i in range(len(cumleler)-1):
        nltkAsdimlari(cumleler[i])
    for i in range(len(cumleler)-1):
        baslikKelimeBul(cumleler[i],basliktakiKelimeler,cumle_uz[i])
    # bertAlgoritmasi(duzenlenmisCumleler)
    label_sayiOzel.config(text=f"Özel İsim skor: {skor_ozel}")
    label_sayi.config(text=f"Numerik skor: {skor_numerik}")
    labelCumleUz.config(text=f"{kelimesay}")
    # for a in range(len(duzenlenmisCumleler)-1):
    #     print(duzenlenmisCumleler[a])

    
    nx.draw(G, with_labels=True)
    plt.show()
def baslikKelimeBul(cuumle,kelimeler,cumleUz):
    a=0
    words = cuumle.lower().split()
    
    for word in words:
        if word in kelimeler:
            a += 1
    a=round(float(a/cumleUz),3)
    kelimesay.append(a)
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
    stemmed_tokens = [stemmer.stem(token) for token in duzenlenmis]  # Her kelimeye stemming işlemi uygulayın
    stemmed_sentence = ' '.join(stemmed_tokens)
    
    duzenlenmisCumleler.append(stemmed_sentence)
def bertAlgoritmasi(duzenlenmis):
    modelAdi = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(modelAdi)
    model = BertModel.from_pretrained(modelAdi)

    input_ids = tokenizer.batch_encode_plus(duzenlenmis, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**input_ids)
    last_hidden_states = outputs.last_hidden_state
    pooled_output = outputs.pooler_output

    similarity_scores = torch.cosine_similarity(pooled_output[0], pooled_output[1:], dim=0)
    i=0
    for i, score in enumerate(similarity_scores):
        print("Cümle {}: {}".format(i+1, duzenlenmis[i]))
        print("Benzerlik skoru: {:.4f}".format(score.item()))
        print()
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


