from tkinter import Label, OptionMenu, StringVar, Tk, Button, filedialog,messagebox, Entry
import main as m
import matplotlib.pyplot as plt

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

buton = Button(root, text="DOSYA SEÇ", command=m.dosya_bul,border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
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
buton2 = Button(root, text="GRAF OLUŞTUR", command=lambda:( m.dosyaKontrol() if m.treshold_degerleri() else None),border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
buton2.pack(pady=10)
buton2.configure(bg="#99627A")
entry = Entry(root,width=80)
entry.pack()
buton3 = Button(root, text="Kıyasla", command=m.textAl,border=5,bd=0,padx=10,pady=5,relief="solid",anchor='ne')
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

ozet_metin = None

root.mainloop()