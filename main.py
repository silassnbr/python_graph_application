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
from tkinter import Label, Tk, Button, filedialog,messagebox, Entry
import re
import string
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import rouge_skor
from transformers import GPT2Tokenizer, GPT2Model
from gensim.models import Word2Vec
from tkinter import *
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
global flag
flag=False
cumle_skor = 0
cumle_benzerlik = 0
p3 =0 
cumleToplamSkor=[]

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
        cumleToplamSkor.clear()
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
def tdfKelimeSkor(duzenlenis,uzunluk):
    puan=0
    duzenlenis=duzenlenis.lower().split()
    for kelime in duzenlenis:
        if kelime in tdfOn:
            puan=puan+1
    puan=round(float(puan/uzunluk),3)
    tdf_skor.append(puan)
    print("tdf sayi")
    print(len(tdf_skor))
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
def vectorBul(sentence, model):
    words = sentence.split()
    vectors = [vectorKelime(word, model) for word in words]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
def vectorKelime(word, model):
    if word in model:
        return model[word]
    else:
        return np.zeros(model.vector_size)

def word2vec(cumle):

    model = KeyedVectors.load('vord2vec.bin')

    # model = Word2Vec.load('GoogleNews-vectors-negative300.bin')

    vector_list = [vectorBul(sentence, model) for sentence in cumle]

    
    similarity_matrix = cosine_similarity(vector_list)


    for i in range(len(cumle)):
        for j in range(i+1, len(cumle)):
            sentence1 = cumle[i]
            sentence2 = cumle[j]
            similarity = similarity_matrix[i][j]
            # print(f"Benzerlik ({sentence1}, {sentence2}): {similarity}")
    G = nx.Graph()
    toplam_kenar = 0
    treshold_gecen = 0
    benzer_node_adet = []

    for i in range(len(cumle)):
        G.add_node(i+1)
        benzer_node_adet.append(0)

    for i in range(len(cumle)):

        for j in range(i+1, len(cumle)):
            
            similarity = similarity_matrix[i][j]
            G.add_edge(i+1, j+1, weight=round(similarity,3))

            if similarity >= cumle_benzerlik:  # Eşik değeri belirleyerek sadece belirli bir benzerlik üzerindeki ilişkileri gösterebilirsiniz
                treshold_gecen += 1
                benzer_node_adet[i] +=1
                benzer_node_adet[j] +=1

            toplam_kenar += 1

    tresholdu_gecen_node(treshold_gecen, toplam_kenar)
    scores = [0.8, 0.5, 0.6, 0.9,0.8,0.6]
    cumleSkorSon()
    node_attributes = {}
    for i, node in enumerate(G.nodes):
        node_attributes[node] = {'size': 300, 'shape': 's', 'score': cumleToplamSkor[i]}

    benzer_node_attributes = {}
    for i, node in enumerate(G.nodes):
        benzer_node_attributes[node] = {'size': 300, 'shape': 's', 'score': benzer_node_adet[i]}    

    pos = nx.circular_layout(G) # Düğümleri konumlandırmak için bir düzen algoritması kullanabilirsiniz
    plt.figure(figsize=(15, 8),facecolor="#99627A") 
    edge_labels = nx.get_edge_attributes(G, "weight")

    for edge, label in edge_labels.items():

        edge_pos = [(pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2]
        
        if label >= cumle_benzerlik:
            bgcolor = "lightgreen"
        else:
            bgcolor = "#99627A"    
        
        plt.text(edge_pos[0], edge_pos[1], label, ha='center', va='center', bbox=dict(facecolor=bgcolor, edgecolor=bgcolor), fontsize=10)

    nx.draw_networkx(G, pos, with_labels=True,node_color="#643843")
   # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color=edge_label_colors )#bbox=dict(facecolor=box_color)

    # Skorları düğümlerin dışına yazdırın
    for node, attr in node_attributes.items():
        x, y = pos[node]
        plt.text(x - 0.10, y, attr['score'], ha='center', va='center',bbox=dict(facecolor="lightcoral", edgecolor="lightcoral"))

    for node, attr in benzer_node_attributes.items():
        x, y = pos[node]
        plt.text(x + 0.05, y, attr['score'], ha='center', va='center',bbox=dict(facecolor="yellow", edgecolor="yellow"))    

    plt.axis('off') 
    plt.show()


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
    toplam_kenar = 0
    treshold_gecen = 0
    benzer_node_adet = []

    for i in range(len(cumlelerSon)):
        G.add_node(i+1)
        benzer_node_adet.append(0)

    for i in range(len(cumlelerSon)):

        for j in range(i+1, len(cumlelerSon)):
            
            similarity = similarity_matrix[i][j]
            G.add_edge(i+1, j+1, weight=round(similarity,3))

            if similarity >= cumle_benzerlik:  # Eşik değeri belirleyerek sadece belirli bir benzerlik üzerindeki ilişkileri gösterebilirsiniz
                treshold_gecen += 1
                benzer_node_adet[i] +=1
                benzer_node_adet[j] +=1

            toplam_kenar += 1

    tresholdu_gecen_node(treshold_gecen, toplam_kenar)
    cumleSkorSon()
    node_attributes = {}
    for i, node in enumerate(G.nodes):
        node_attributes[node] = {'size': 300, 'shape': 's', 'score': cumleToplamSkor[i]}

    benzer_node_attributes = {}
    for i, node in enumerate(G.nodes):
        benzer_node_attributes[node] = {'size': 300, 'shape': 's', 'score': benzer_node_adet[i]}    

    pos = nx.circular_layout(G) # Düğümleri konumlandırmak için bir düzen algoritması kullanabilirsiniz
    plt.figure(figsize=(15, 8),facecolor="#99627A") 
    edge_labels = nx.get_edge_attributes(G, "weight")

    for edge, label in edge_labels.items():

        edge_pos = [(pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2]
        
        if label >= cumle_benzerlik:
            bgcolor = "lightgreen"
        else:
            bgcolor = "#99627A"    
        
        plt.text(edge_pos[0], edge_pos[1], label, ha='center', va='center', bbox=dict(facecolor=bgcolor, edgecolor=bgcolor), fontsize=10)

    nx.draw_networkx(G, pos, with_labels=True,node_color="#643843")
   # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color=edge_label_colors )#bbox=dict(facecolor=box_color)

    # Skorları düğümlerin dışına yazdırın
    for node, attr in node_attributes.items():
        x, y = pos[node]
        plt.text(x - 0.10, y, attr['score'], ha='center', va='center',bbox=dict(facecolor="lightcoral", edgecolor="lightcoral"))

    for node, attr in benzer_node_attributes.items():
        x, y = pos[node]
        plt.text(x + 0.05, y, attr['score'], ha='center', va='center',bbox=dict(facecolor="yellow", edgecolor="yellow"))    

    plt.axis('off') 
    plt.show()
def cumleSkorSon():
    for i in range(len(skor_ozel)):
        sonSkor=(2*skor_ozel[i])+skor_numerik[i]+kelimesay[i]*2+tdf_skor[i]*3
        cumleToplamSkor.append(sonSkor)
    print(cumleToplamSkor)
def tresholdu_gecen_node(treshold_gecen, toplam_kenar):
    print(treshold_gecen, toplam_kenar)
    global p3
    p3 = treshold_gecen/ toplam_kenar
    print("p3", p3)

def baslikKelimeBul(cuumle,kelimeler,cumleUz):
    a=0
    words = cuumle.lower().split()
    
    for word in words:
        if word in kelimeler:
            a += 1
    a=round(float(a/cumleUz),3)
    kelimesay.append(a)
    print("kelime baslık")
    print(len(kelimesay))
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
    
def skorDonustur(ozel,cumle,numer):
    a=round(float(ozel/cumle),3)
    skor_ozel.append(a)
    print("ozel skor")
    print(len(skor_ozel))
    b=round(float(numer/cumle),3)
    skor_numerik.append(b)
    print("numerik skor")
    print(len(skor_numerik))
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
    secim=secilenAlgoritma.get()
    if(secim=='Word2vec'):
        print(secim)
        if (flag==True):
            for i in range(len(cumleler)-1):
                baslikKelimeBul(cumleler[i],basliktakiKelimeler,cumle_uz[i])
            metin = " ".join(duzenlenmisCumleler)
            kelimesay=metin.split()
            sayisi=len(kelimesay)
            tdfDegerBulma(duzenlenmisCumleler,sayisi)
            for i in range(len(duzenlenmisCumleler)):
                tdfKelimeSkor(duzenlenmisCumleler[i],cumle_uz[i])
            word2vec(duzenlenmisCumleler)
            
        
        else:
            messagebox.showinfo("UYARI","Dosya seçmediniz")
    else:
        if (flag==True):
            for i in range(len(cumleler)-1):
                baslikKelimeBul(cumleler[i],basliktakiKelimeler,cumle_uz[i])
            
            metin = " ".join(duzenlenmisCumleler)
            kelimesay=metin.split()
            sayisi=len(kelimesay)
            tdfDegerBulma(duzenlenmisCumleler,sayisi)
            for i in range(len(duzenlenmisCumleler)):
                tdfKelimeSkor(duzenlenmisCumleler[i],cumle_uz[i])
            gloveDeneme(duzenlenmisCumleler) 
              
        else:
            messagebox.showinfo("UYARI","Dosya seçmediniz")

def treshold_degerleri():

    global cumle_benzerlik
    global cumle_skor

    if entry1.get() == "" or entry2.get() == "":

        if(entry1.get() != ""):
            cumle_benzerlik = float(entry1.get())
        if(entry2.get() != ""):
            cumle_skor = float(entry2.get())    

        messagebox.showinfo("UYARI","Treshold değerlerini giriniz")
        return False #False a çevrilecek
    
    else:    
        cumle_benzerlik = float(entry1.get())
        cumle_skor = float(entry2.get())
        print("cumle_benzerlik 1:", cumle_benzerlik , "cumle_skor 2:", cumle_skor)
        return True
def textAl():
    candidate_text=' '.join(duzenlenmisCumleler)
    text = entry.get()  
    print(text)
    if text is not None:
        rouge_skor.rouge_scores(candidate_text, text)
    
        SKOR= []
        r, p, f = rouge_skor.calculate_rouge_1(candidate_text, text)
        SKOR.append([r,p,f])
        r, p, f = rouge_skor.calculate_rouge_2(candidate_text, text)
        SKOR.append([r,p,f])
        r, p, f = rouge_skor.calculate_rouge_l(candidate_text, text)
        SKOR.append([r,p,f])

        label_baslik.configure(text="       r       p       f")
        label3.configure(text=f"rouge-1 {SKOR[0]}")
        label4.configure(text=f"rouge-2 {SKOR[1]}")
        label5.configure(text=f"rouge-l {SKOR[2]}")

        print(SKOR)
    else:
        messagebox.showinfo("UYARI","Lütfen kıyaslamak için metin giriniz!!!")

root = Tk()
root.title("Dosya Seçme Uygulaması")
root.configure(bg="#C88EA7")
yaziboyutu=("Arial",16)
etiket = Label(root, text="LÜTFEN DOSYA SEÇİNİZ",font=yaziboyutu,fg="#643843",anchor='w')
etiket.pack(pady=10)
etiket.configure(bg="#C88EA7")

pencere_genislik = 800
pencere_yukseklik = 600
ekran_genislik = root.winfo_screenwidth()
ekran_yukseklik = root.winfo_screenheight()
x_konumu = int((ekran_genislik - pencere_genislik) / 2)
y_konumu = int((ekran_yukseklik - pencere_yukseklik) / 2)
root.geometry(f"{pencere_genislik}x{pencere_yukseklik}+{x_konumu}+{y_konumu}")

buton = Button(root, text="DOSYA SEÇ", command=dosya_bul,border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
buton.pack(pady=0)
buton.configure(bg="#99627A")

label1 = Label(root, text="Cümle Benzerliği Tresholdu:", fg="#643843")
label1.configure(bg="#C88EA7")
label1.pack(pady=15)
entry1 = Entry(root)
entry1.pack(pady=0)

label2 = Label(root, text="Cümle Skoru Tresholdu:", fg="#643843")
label2.configure(bg="#C88EA7")
label2.pack(pady=15)
entry2 = Entry(root)
entry2.pack(pady=0)


label_sayiOzel = Label(root, text="",fg="#643843")
label_sayiOzel.pack(pady=5)
label_sayiOzel.configure(bg="#C88EA7")

#dropdown
algoritmalar = ["Word2vec", "Glove"]

secilenAlgoritma = StringVar(root)
#Vord2vec varsayılan olarak ayarlandı
secilenAlgoritma.set(algoritmalar[0])  
dropdown = OptionMenu(root, secilenAlgoritma, *algoritmalar)
dropdown.pack(pady=5)
dropdown.config(font=("Arial", 8))
dropdown.configure(font=5,foreground="#99627A",background="white")
dropdown.pack()

########
buton2 = Button(root, text="GRAF OLUŞTUR", command=lambda:( dosyaKontrol() if treshold_degerleri() else None),border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
buton2.pack(pady=10)
buton2.configure(bg="#99627A")
entry = Entry(root,width=80)
entry.pack()
buton3 = Button(root, text="Kıyasla", command=textAl,border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
buton3.pack(pady=10)
buton3.configure(bg="#99627A")

label_baslik = Label(root, text=" ", fg="#643843")
label_baslik.configure(bg="#C88EA7")
label_baslik.pack(pady=3)
label3 = Label(root, text="Rouge Skoru r", fg="#643843")
label3.configure(bg="#C88EA7")
label3.pack(pady=3)
label4 = Label(root, text="Rouge Skoru p", fg="#643843")
label4.configure(bg="#C88EA7")
label4.pack(pady=3)
label5 = Label(root, text="Rouge Skoru f", fg="#643843")
label5.configure(bg="#C88EA7")
label5.pack(pady=3)

root.mainloop()


