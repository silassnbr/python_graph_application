import random
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import os
from nltk.chunk import ne_chunk
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer, GPT2Model
# dosya seçme fonksiyonu 
ozel_isimler = []
sayilar=[]
cumle_uz=[]
kelimesay=[]
skor_ozel=[]
skor_numerik=[]
duzenlenmisCumleler=[]
tdf_skor=[]
tdfOn=[]
baglantiBir=[]
baglantiIki=[]
benzerlik=[]
global flag
flag=False
def dosya_bul():
    global flag
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
        tdfOn.clear()
        tdf_skor.clear()
        baglantiBir.clear()
        baglantiIki.clear()
        if txt_kontrol(dosya_yolu):
            flag=True
            node_olustur(dosya_yolu)
        else:
            flag=False
            label_sayiOzel.config(text=f"Dosya Seçilemedi")
            messagebox.showinfo("UYARI","Lütfen txt formatında bir dosya seçiniz")
    else:
        flag=False
        label_sayiOzel.config(text=f"Dosya Seçilemedi")
        messagebox.showinfo("UYARI","Dosya Seçilemedi")

#node oluşturma fonksiyonu
def node_olustur(dosya_yolu):
    global flag
    with open(dosya_yolu, encoding="utf8") as dosya:
        dosya_icerigi = dosya.read()
        satirlar=dosya_icerigi.split('\n')
        baslik=satirlar[0].strip()
        metin = '\n'.join(satirlar[1:]).strip()
        global cumleler
        cumleler = metin.split(".")
    
    global basliktakiKelimeler
    basliktakiKelimeler=baslik.lower().split()
    for i in range(len(cumleler)-1):
        ozel_isimSay(cumleler[i])
        numerikSayisi(cumleler[i])
        cumleUzunlugu(cumleler[i])
        skorDonustur(ozel_isimler[i],cumle_uz[i],sayilar[i])

    for i in range(len(cumleler)-1):
        nltkAsdimlari(cumleler[i])
    label_sayiOzel.config(text=f"Dosya Seçildi")
    # G = nx.Graph()
    # # graph olustruma kısmı DÜZENLENECEK###############################
    # for i in range(len(cumleler)-1):
    #     G.add_node(cumleler[i],label=cumleler[i])

    # for i in range(len(baglantiBir) - 1):
    #     G.add_edge(baglantiBir[i],baglantiIki[i])
    # #########################


    # nx.draw(G, with_labels=True)
    # plt.show()
def gpt_deneme(sentences):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2Model.from_pretrained("gpt2-medium")

   

# Cümleleri GPT-3 ile kodlayın
    encoded_sentences = [tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt") for sentence in sentences]

# Cümlelerin temsillerini alın
    representations = []
    with torch.no_grad():
        for encoded_sentence in encoded_sentences:
            representation = model(encoded_sentence)[0].squeeze()
            representations.append(representation)

# Kozinüs benzerliğini hesapla
    similarity_scores = cosine_similarity(representations)

# Benzerlik skorlarını yazdır
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_score = similarity_scores[i, j]
            print(f"Benzerlik Skoru ({sentences[i]} - {sentences[j]}): {similarity_score}")
def tdfKelimeSkor(duzenlenis,uzunluk):
    puan=0
    duzenlenis=duzenlenis.lower().split()
    for kelime in duzenlenis:
        if kelime in tdfOn:
            puan=puan+1
    puan=round(float(puan/uzunluk),3)
    tdf_skor.append(puan)
def tdfDegerBulma(duzenli,sayi):
  
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(duzenli)

    num_documents, num_features = tfidf_matrix.shape
    buyukBul=[]
    buyukBulKelime=[]
# Tdf degeri 0 dan buyuk olanalrı tut
    for i in range(num_documents):
        # print(f"Metin Belgesi {i+1}:")
        feature_names = vectorizer.get_feature_names_out()
        for j in range(num_features):
            word =feature_names[j]
            tfidf = tfidf_matrix[i, j]
            # print(f"Kelime: {word}, TF-IDF: {tfidf}")
            if(tfidf>0):
                buyukBul.append(tfidf)
                buyukBulKelime.append(word)
        # print()
    # for i in range(len(buyukBul)):
        
    #     print(f"{buyukBulKelime[i]}  --  {buyukBul[i]}")
    # print("**********************************")
    sayi=int(sayi*10/100)
    en_buyuk_indeksler = np.argsort(buyukBul)[-sayi:]
    for i in range(len(en_buyuk_indeksler)):
        sira=en_buyuk_indeksler[i]
        tdfOn.append(buyukBulKelime[sira])
def gloveDeneme(cumlelerSon):
    glove_model = KeyedVectors.load('model.bin')
        
    sentence_vectors = []
    vectors=[]
    # Her cümle için vektörleri hesaplayın ve listeye ekleyin
    for sentence in cumlelerSon:
    # Cümleyi kelimelere ayırın
        words = sentence.lower().split()
        for word in words:
            if word in glove_model.key_to_index:
                vectors.append(glove_model.get_vector(word))
            else:
                similar_word = glove_model.most_similar(word)[0][0]
                similar_vector = glove_model.get_vector(similar_word)
                vectors.append(similar_vector)
                
        sentence_vector = sum(vectors)

        sentence_vectors.append(sentence_vector)

# Cümleler arası benzerlık hesabı
    similarity_matrix = np.zeros((len(cumlelerSon), len(cumlelerSon)))
    for i in range(len(cumlelerSon)):
        for j in range(i+1, len(cumlelerSon)):
            similarity = cosine_similarity([sentence_vectors[i]], [sentence_vectors[j]])[0][0]
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
            # print(f"Benzerlik ({i+1} <-> {j+1}): {similarity}")
    G = nx.Graph()
    for i in range(len(cumlelerSon)):
        G.add_node(i+1)
    for i in range(len(cumlelerSon)):
        for j in range(i+1, len(cumlelerSon)):
            similarity = similarity_matrix[i][j]
            if similarity > 0.5:  # Eşik değeri belirleyerek sadece belirli bir benzerlik üzerindeki ilişkileri gösterebilirsiniz
                G.add_edge(i+1, j+1, weight=round(similarity,3))
    scores = [0.8, 0.5, 0.6, 0.9,0.8,0.6]
    node_attributes = {}
    for i, node in enumerate(G.nodes):
        node_attributes[node] = {'size': 300, 'shape': 's', 'score': scores[i]}

    pos = nx.circular_layout(G) # Düğümleri konumlandırmak için bir düzen algoritması kullanabilirsiniz
    plt.figure(figsize=(20, 20),facecolor="#99627A") 
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx(G, pos, with_labels=True,node_color="#643843")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # Skorları düğümlerin dışına yazdırın
    for node, attr in node_attributes.items():
        x, y = pos[node]
        plt.text(x - 0.05, y, attr['score'], ha='center', va='center')
    plt.axis('off') 
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
   
    punct_sentence = ''.join(char for char in duzenle if char not in punctuation)
  
    
    
    stop_words = set(stopwords.words("english"))  
    stop_tokens = word_tokenize(punct_sentence)  # Cümleyi kelimelere ayırın
    filtered_tokens = [token for token in stop_tokens if token.lower() not in stop_words]  
    filtered_sentence = ' '.join(filtered_tokens) 

    # Her kelimeye stemming işlemi uygulayın
    duzenlenmis = sent_tokenize(filtered_sentence)
    stemmed_tokens = [stemmer.stem(token) for token in duzenlenmis]  
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

def dosyaKontrol():
    if (flag==True):
        print("graf olustur")
        for i in range(len(cumleler)-1):
            baslikKelimeBul(cumleler[i],basliktakiKelimeler,cumle_uz[i])
        # bertAlgoritmasi(duzenlenmisCumleler)
        metin = " ".join(duzenlenmisCumleler)
        kelimesay=metin.split()
        sayisi=len(kelimesay)
        gloveDeneme(duzenlenmisCumleler)
        # gpt_deneme(duzenlenmisCumleler)
        tdfDegerBulma(duzenlenmisCumleler,sayisi)
        for i in range(len(duzenlenmisCumleler)-1):
            tdfKelimeSkor(duzenlenmisCumleler[i],cumle_uz[i])
        
        # label_sayi.config(text=f"Numerik skor: {tdf_skor}")
        # labelCumleUz.config(text=f"{tdf_skor}")    
    else:
        messagebox.showinfo("UYARI","Dosya seçmediniz")
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

# label_sayiOzel = Label(root, text="",fg="#643843")
# label_sayiOzel.pack(pady=10)
# label_sayiOzel.configure(bg="#C88EA7")

# label_sayi = Label(root, text="",fg="#643843")
# label_sayi.pack(pady=10)
# label_sayi.configure(bg="#C88EA7")

# labelCumleUz = Label(root, text="",fg="#643843")
# labelCumleUz.pack(pady=10)
# labelCumleUz.configure(bg="#C88EA7")

label_sayiOzel = Label(root, text="",fg="#643843")
label_sayiOzel.pack(pady=10)
label_sayiOzel.configure(bg="#C88EA7")

buton2 = Button(root, text="GRAF OLUŞTUR", command=dosyaKontrol,border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
buton2.pack(pady=10)
buton2.configure(bg="#99627A")
root.mainloop()


