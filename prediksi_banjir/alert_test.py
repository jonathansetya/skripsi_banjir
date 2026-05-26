from notification import send_notification

print("=== SIMULASI PERINGATAN BANJIR ===")
print("1. KODE ORANYE (24 Jam)")
print("2. KODE MERAH (2 Jam)")
print("3. KODE KRITIS (1 Jam)")

pilih = input("Pilih simulasi: ")

if pilih == "1":

    send_notification(
        "⚠️ KODE ORANYE",
        "Potensi banjir dalam 24 jam. Harap meningkatkan kewaspadaan."
    )

    print("Notifikasi ORANYE berhasil dikirim")

elif pilih == "2":

    send_notification(
        "🚨 KODE MERAH",
        "Potensi banjir dalam 2 jam. Segera lakukan persiapan darurat."
    )

    print("Notifikasi MERAH berhasil dikirim")

elif pilih == "3":

    send_notification(
        "🆘 STATUS KRITIS",
        "Potensi banjir dalam 1 jam. Segera lakukan evakuasi."
    )

    print("Notifikasi KRITIS berhasil dikirim")

else:
    print("Pilihan tidak valid")