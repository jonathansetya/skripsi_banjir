from notification import send_notification


# =========================
# FUNCTION ALERT
# =========================
def check_alert(status):

    if status == "ORANYE":

        success = send_notification(
            "⚠️ KODE ORANYE",
            "Potensi banjir dalam 24 jam. Harap meningkatkan kewaspadaan."
        )

        if success:
            print("Notifikasi ORANYE berhasil dikirim")
        else:
            print("Notifikasi ORANYE gagal dikirim")

    elif status == "MERAH":

        success = send_notification(
            "🚨 KODE MERAH",
            "Potensi banjir dalam 2 jam. Segera lakukan persiapan darurat."
        )

        if success:
            print("Notifikasi MERAH berhasil dikirim")
        else:
            print("Notifikasi MERAH gagal dikirim")

    elif status == "KRITIS":

        success = send_notification(
            "🆘 STATUS KRITIS",
            "Potensi banjir dalam 1 jam. Segera lakukan evakuasi."
        )

        if success:
            print("Notifikasi KRITIS berhasil dikirim")
        else:
            print("Notifikasi KRITIS gagal dikirim")

    else:
        print("Status tidak valid")


# =========================
# TEST MANUAL
# =========================
if __name__ == "__main__":

    print("=== SIMULASI PERINGATAN BANJIR ===")
    print("1. KODE ORANYE (24 Jam)")
    print("2. KODE MERAH (2 Jam)")
    print("3. KODE KRITIS (1 Jam)")

    pilih = input("Pilih simulasi: ")

    if pilih == "1":
        check_alert("ORANYE")

    elif pilih == "2":
        check_alert("MERAH")

    elif pilih == "3":
        check_alert("KRITIS")

    else:
        print("Pilihan tidak valid")