from fastapi.responses import PlainTextResponse
from fastapi import APIRouter
import random

router = APIRouter()


@router.get("/")
def motd_root():
    return PlainTextResponse(random_motd())


@router.get("/health")
@router.get("/healthz")
def health_endpoint():
    """Health and diagnostics endpoint."""
    from pathlib import Path
    from server.config import DEFAULT_DB_PATH
    db_path = Path(DEFAULT_DB_PATH)
    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    error = None
    count = 0
    db_ok = False
    try:
        from server.services.stash import get_stash
        stash = get_stash()
        with stash.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM storedcrawlresult")
            count = cursor.fetchone()[0]
            db_ok = True
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return {
        "status": "ok" if db_ok else "degraded",
        "db_path": str(db_path),
        "db_exists": db_exists,
        "db_size_bytes": db_size,
        "row_count": count,
        "error": error,
    }


def random_motd():
    return (
        random.choice(MESSAGES)
        + "\n\n"
        + random.choice(RANDOM_CHARACTERS)
        + "\n\n(reload for random messages)"
    )

MESSAGES = [
    """    Thằng Bờm có cái quạt mo
Tú Ông xin đổi khô gà, mùn cưa
    Bờm rằng: Bờm chẳng lấy cưa
Tú ông xin đổi một vựa sisa
    Bờm rằng: Bờm chẳng lấy sa
Tú ông xin đổi người cha tuyệt vời
    Bờm rằng: Bờm chẳng lấy vời
Tú ông xin đổi một đời nghiệp binh""",
    """trước mình đánh cờ với Quang Liêm.
Trong vòng 20 nước cờ Liêm đã thua mình.
Liêm hỏi: "anh là ai mà sao cao cờ quá vậy, 
anh thậm chí có thể đứng trong top 7 thế giới".
Mình bảo: "thật ra mình thuộc top 3, nhưng mình 
ko muốn lộ diện, Liêm đánh rất hay, cố gắng lên nhé".
Liêm không nói gì, gục xuống, trong khi mình bước 
đi không quay lại
""",
    """❗SỐC NẶNG❗: PHÙNG KHÁNH LINH ĐI NGỦ RỒI TỈNH DẬY !?

Mới đây cộng đồng mạng hoảng hốt khi biết tin nữ ca sĩ Phùng 
Khánh Linh bất ngờ tỉnh dậy trên giường của mình.
Cô chia sẻ: "tối đó tôi ngủ rồi sáng nay tôi tỉnh dậy". 
Cô trở thành tâm điểm bàn luận trên mạng xã hội. Mọi người còn 
ca ngợi cô là "tấm gương vượt khó".

Bạn nghĩ sao về điều này ?
""",
    """Kính gửi Quý khách hàng,
VinFast chân thành cảm ơn Quý khách đã tin tưởng sử dụng sản phẩm.
Quý khách đã mua xe tại VinFast DTH vào 12:00 AM ngày 5/6/2026. 
Vui lòng truy cập đường link sau để xác nhận thông tin bảo hành: 
https://vinfastauto.com/vn_vi/so-bao-hanh-dien-tu?token=3iFISF38A3q9fajsfsvawrh
""",
    """THÔNG BÁO TUYỂN SINH ĐẠI HỌC ROBLOX NĂM HỌC 2026

🔥 CƠ HỘI TRỞ THÀNH SINH VIÊN ĐẠI HỌC ROBLOX chính thức 
mở cổng đăng ký xét tuyển cho khóa mới. 
Nếu bạn có niềm đam mê bất tận với Roblox và lập trình, 
Đại học Roblox là lựa chọn hàng đầu!

📊 CHỈ TIÊU TUYỂN SINH

Tổng chỉ tiêu: 1.000 học sinh Ưu tiên nạp Robux.
""",
    """Hút thuốc lào nâng cao sĩ diện.
Thơm mồm bổ phổi, diệt trùng lao.
Nâng điếu lên như Triệu Tử cầm đao.
Nhả khói ra như Khổng Minh gọi gió.

Một thằng hút, bốn thằng say
Thằng con châm điếu lăn quay ra nhà,
Có chị hàng xóm đi qua,
Hít phải khói thuốc cũng say ba bốn tuần.
""",
"""
                     cos(2x) - 1
Tính giới hạn  lim  ─────────────
               x→0    ln(1 - x)
                 
Sử dụng Khai triển Maclaurine:
-  cos(2x)   = 1 - 2x² + o(x²)
-  ln(1 - x) = -x - (x²/2) + o(x²)

Thay vào biểu thức giới hạn:
             -2x² + o(x²)              -2x + o(x)             0
   lim   ───────────────────── = lim ──────────────────── = ───── = 0
   x→0    -x - (x²/2) + o(x²)    x→0   -1 - (x/2) + o(x)     -1
""",
"""Nhà bỗng hôm nay vắng tiếng cười 
Cha về độ ấy nét buồn hơi 
Sừng sững tay cầm roi chi bảo
Quật xuống âm vang vọng đất trời
""",
"""hey siri
🤖: Vâng bệ hạ
Gọi tất cả bạn bè của tôi
🤖: LỖI: Không có bạn bè được phát hiện
Ý tôi là công cụ của tôi
🤖: Đã hiểu thưa bệ hạ. Gọi cho 8 tỷ người…
Chiếu tướng.
""",
"""Hồi chiêù s ko di học??
Hồi hồi.... Chiều...hồi m..ư..a.....hồi ... Hồi chiều..... Mưa ..... Hồi.... chiều....mưa ..... thật thật
""",
"""
    A en sán sàn
đi chơi côn viên lã làn
    Côn viên lã làn
Ở khu côn nghiệt bát nin
    Ở đây tật chun
Điên viên ai đồ nổi tiến
    Có tô ngộ khôn
Yêu quái cao thủ vó lân

    Côn viên lã làn
Khôn có người bát nin đô
    En ra đây chơi
A mua cho en cốt nướt
    Ko có tiền trạ
Mua chị mua chị cho en
    
    A sẽ cho en
Đi ra côn viên lá làn
    Xen ai đồ nhể
Ai đồ nhẩy điệu con ết
    En cú nhảy đi
Nhảy đi cho a xem vớ 
    Em nhảy như đi
Em nhảy em nhảy như điền
    
    Em muốn ghét đôi
Em muốn em muốn ghét đôi
    Em ghét đôi
Ở trong côn viên lã làn
    Côn viên lã làn
Có nhiều trai xinh gái đệt
    Muốn có người iêu
Hãy da côn viên lã làn
""",
"""
        BỘ GIÁO DỤC VÀ ĐÀO TẠO           CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
    TRƯỜNG ĐẠI HỌC ROBLOX VIỆT NAM          Độc lập - Tự do - Hạnh phúc
        ──────────────────────              ───────────────────────────

                                         TP. Bá Khí, ngày 6 tháng 7 năm 2026

                                  THÔNG BÁO
            V/v quản lý “bá khí” chạy ra giữa sân trường khi trời mưa


        Kính gửi: Toàn thể sinh viên của Nhà trường
                  Ủy ban Bá khí Việt Nam

        Nhà trường luôn khuyến khích học sinh vui chơi, vận động và tham gia 
    các hoạt động tập thể một cách vui vẻ, lành mạnh. Tuy nhiên, nhà trường
    nghiêm cấm học sinh đu trend “bá khí”, chạy ra giữa sân trường khi trời 
    mưa, cũng như thực hiện các hành vi chạy nhảy, đu bám hoặc vui chơi nguy
    hiểm trong điều kiện thời tiết không đảm bảo.

        Việc chạy ra giữa sân trường khi trời mưa có thể khiến sân trơn 
    trượt, dễ té ngã, va chạm, cảm lạnh hoặc ảnh hưởng đến sức khỏe. Chỉ
    một phút bất cẩn cũng có thể làm gián đoạn việc học tập, khiến gia
    đình lo lắng và ảnh hưởng đến các hoạt động chung của lớp, của trường.

        Nhà trường không cấm các em vui chơi, mà luôn mong muốn các em có
    những giờ phút vui vẻ, thoải mái và an toàn. Các em hãy lựa chọn những hoạt
    động phù hợp, thực hiện theo hướng dẫn của thầy cô và không chạy ra ngoài
    khi trời mưa hoặc thời tiết quá nắng nóng.

    Nhà trường đề nghị tất cả học sinh nghiêm túc chấp hành:
      - Không đu trend “bá khí” chạy ra giữa sân trường khi trời mưa.
      - Không chạy nhảy, đùa nghịch trên sân ướt hoặc khu vực trơn trượt.
      - Khi trời mưa, cần ở trong khu vực an toàn theo hướng dẫn của thầy cô.
      - Biết nhắc nhở bạn bè cùng giữ gìn an toàn.

    Vui chơi là cần thiết, nhưng an toàn phải được đặt lên hàng đầu.
    Hãy vui chơi đúng cách, vui khỏe, an toàn và không ảnh hưởng đến việc học tập!

    Nơi nhận:                                            T/M BAN GIÁM HIỆU
     - Như trên                                             Hiệu trưởng
     - Lưu VP                                                
                                                              (đã ký)

                                                             Robert Lox
"""
]

RANDOM_CHARACTERS = [
    r"""
⣿⣿⣿⣿⡿⠿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⠋⣴⣾⠛⡆⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠐⢻⢿⣦⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⡇⢸⢯⡎⢸⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⡾⢸⣷⢹⡇⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣷⠘⡞⠍⠈⢻⣀⢛⣉⣭⣥⣤⣤⣤⣭⣉⣋⣰⡁⢀⣞⣼⠃⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣷⡌⠛⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⡟⢁⣼⡿⠛⣛⡛⢩⣛⡻⠿⣿⣿⣿
⣿⣿⣿⡟⢡⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡙⢾⡄⣿⣇⣾⣏⣠⣮⢻⣿⣿
⣿⣿⡟⢠⣿⣿⣿⣿⠛⣩⣍⠂⣼⢟⣟⢻⡧⢈⣭⣍⠻⣿⣿⣿⣷⡈⣿⣿⣿⣿⣿⣿⣋⣁⠛⢿
⣯⡤⢶⣾⣿⣿⣿⣇⢀⣙⣉⠀⣼⠻⣿⢇⡇⢈⣉⣁⢀⣿⣿⣿⣿⣧⣬⡉⣹⣿⣿⣿⡿⢿⣼⣿
⣥⡶⢿⣿⡿⠋⡹⠫⠔⣤⣤⣶⡟⠿⠉⠿⢻⣶⣦⣶⠚⠏⡏⡝⢿⣿⡧⣅⢻⣿⣿⣿⣶⣦⢈⣿
⣏⣰⢿⣿⣧⣪⠴⠤⣁⣾⣿⣿⣿⣖⣛⣲⣿⣿⣿⣧⣘⣰⣴⣣⣿⣿⠷⢈⣿⣿⣿⣿⣬⡙⢾⣿
⣿⣇⣞⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢹⣆⣿⣿⣿⣿⡙⠻⣹⣿⣿
⣿⣿⣿⡇⢨⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⡄⢾⣿⣿⣿⣿⠻⡿⣠⣿⣿⣿
⣿⣿⣿⡇⢸⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢸⠇⣸⣿⣿⡟⢿⡧⣰⣿⣿⣿⣿
⣿⣿⣿⣿⣤⡁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⢠⣴⣿⡏⢹⣇⣨⣾⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢀⡌⢸⣧⣴⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⠀⣿⠉⣛⠛⠛⠛⠻⠿⠟⠛⠛⢛⣛⠋⣹⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣇⠛⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠛⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
""",
    r"""
⠀⠀⠀⠀⠀⠀⢀⡞⠹⣦⠰⡞⠙⣆⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢴⠀⠀⣿⠐⡇⠀⢻⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢸⠀⠀⣿⢸⡇⠀⢸⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⡸⣄⠀⣿⣨⡇⠀⣟⡀⠀⠀⠀⠀⠀
⠀⠀⢠⡶⠚⠉⢁⡀⠀⠀⠀⠀⠀⡈⠉⠙⠲⣤⡀⠀
⢀⡶⠋⠀⢀⠔⠉⠀⠀⠀⠀⠀⠀⠈⠑⢄⠀⠈⠻⡄
⣾⠁⠀⠀⠈⠀⣠⣂⡄⠀⠀⠀⣔⣢⠀⠈⠀⠀⠀⢹
⡇⠀⠀⢠⣠⣠⡌⠓⠁⠀⡀⠀⠙⠊⡄⢀⣀⠀⠀⢸
⢷⡀⠀⠈⠁⠁⠀⠀⠈⠓⡓⠂⠀⠀⠉⠈⠁⠀⠀⡼
⠈⠳⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡞⠁
⠀⠀⢾⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⢸⠀⠀
⠀⠀⠈⢻⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡾⠊⠁⠀
⠀⠀⠀⠘⣇⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⡀⣷⠀⠀⠀
⠀⠀⠀⠀⢿⣼⠉⠉⠙⠛⠛⠛⠛⠉⢹⣁⠟⠀⠀⠀
""",
    r"""
⠀⠀⠀⠀⠀⣴⣦⡀⠀⠀⠀⠀⠀⠀⠀⣠⣤⡄⠀⠀⠀⠀
⠀⠀⠀⠀⣾⣏⢾⣹⣦⡴⣶⢦⣤⣤⣾⢻⡵⣻⡆⠀⠀⠀
⠀⠀⠀⣴⣿⡜⣧⣳⣮⡷⠛⠻⣼⣣⢏⡷⣹⢳⣇⠀⠀⠀
⠀⡴⠃⠀⠈⠉⠉⢉⠀⠀⠀⠀⢀⠉⠛⠓⠛⠛⠉⠙⣦⠀
⣸⠁⠀⠀⠀⠀⣶⢳⡆⠀⠀⠀⢰⣒⣦⠀⠀⠀⠀⠀⠘⣆
⣿⠀⠀⠀⢾⡶⢭⠉⠩⣀⣆⣠⠈⠉⡽⣔⣦⠀⠀⠀⠀⣿
⢻⡄⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡏
⠀⠻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠀
⠀⠀⢛⠩⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣵⠀⠀
⠀⠀⠀⠙⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠛⠁⠀⠀
⠀⠀⠀⠀⠹⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠷⣴⣄⠀⠀
⠀⠀⠀⠀⠀⠀⠓⠦⣤⣀⢠⠀⣠⣤⠶⠋⠛⠛⠗⠋⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠠⠦⠽⠾⠶⠧⠦⠄⠀⠀⠀⠀⠀⠀⠀
""",
    r"""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠞⢢⡀⠀⠀⣱⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⡿⡇⠁⠀⠈⠰⣶⣗⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠄⠊⠘⠛⠋⠀⠀⠀⠀⠀⠈⠛⠈⠦⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⣰⠀⠀⠀⠀⠀⠀⠀⠈⡀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⠄⠛⠀⠀⠀⠀⠞⢦⠷⢦⠀⠀⠀⠀⠀⡰⡁⠀⠀⠀⠀
⠀⠀⢀⡰⠁⠀⠀⠀⠀⣀⢠⠤⡤⡠⢤⡀⠀⠀⠀⠈⠀⠐⠆⠀⠀⠀
⠀⢀⠎⠀⠀⠀⠀⠐⠠⡈⠏⣿⢱⣛⣳⠈⠀⡠⠀⠀⠀⠀⠈⡄⠀⠀
⠀⡜⠀⠀⠀⠀⠀⠀⠀⠱⠀⠀⠀⠀⠀⠀⡌⠀⠀⠀⠀⠀⠀⠘⠄⠀
⢴⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠀⠤⠄⠀⠊⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀
⡇⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⢀⠄⠀⠀⠀⠀⠀⠀⠀⠀⢡
⠏⠀⠀⠀⠀⠀⠀⠀⠑⠃⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠠
""",
    r"""
        ^__^
        (oo)\_______
        (__)\       )\/\
            ||----w |
            ||     ||
""",
    r"""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠒⣆⡤⣄⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡌⠀⢠⡏⠀⢸⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣰⠀⠀⣸⠁⠀⡎⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣠⠞⠉⣁⡀⠀⠘⠃⠀⠻⣤⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⡰⠋⢀⡼⠉⠀⠀⠀⠀⠀⠉⠉⢲⡈⠳⡄⠀⠀⠀⠀
⠀⠀⠀⠎⠀⠀⠈⠀⣴⣦⠀⠀⠀⣠⣄⠀⠀⠁⠀⠀⢳⡀⠀⠀
⠀⠀⢸⠀⠀⠰⠢⠀⠙⠉⠀⠀⠀⠙⠚⠀⡀⠀⠀⠀⠀⡇⠀⠀
⠀⠀⢸⠀⠀⠀⠀⠀⠀⠐⢈⢛⠡⠄⠀⠈⠁⠃⠀⠀⠀⡇⠀⠀
⠀⡞⠙⢦⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⣠⣇⡀⠀
⠀⢈⡦⠀⠀⠀⠀⠀⠀⠐⠦⠖⠀⠀⠀⠀⠀⠀⠀⠀⠋⢀⠃⠀
⢀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠋⠀⠀
⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⠀⠀
⠣⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀
⠀⣏⢠⣤⣀⡀⠀⠀⠀⢀⠄⠀⠀⠀⠀⠀⠀⠀⠀⢀⠌⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠉⠉⠉⠙⠤⠞⠉⠉⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀
""",
    """
⡀⢂⠐⠈⠄⠂⡐⠠⢈⠠⠐⢈⠠⡕⢆⣰⠜⠖⠕⠴⢄⣂⠤⡄⠂⡁⠄⠂⡁⠄⠂⡁⠄⠂⡈⢐⠀⠂⠐⠀⡂⠨⠀⠅
⢀⠂⡈⠄⡁⢂⠐⡀⢂⠐⡈⣠⣘⠄⠈⠀⠀⠀⠀⠀⠀⠁⠰⠉⡇⡀⢂⠁⠄⠂⡁⠄⠂⡁⠐⠀⢀⠐⠀⡀⠐⡈⠠⠁
⠄⠂⡐⠠⠐⢀⠂⡐⣠⠞⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠓⡄⠂⠄⡁⢂⠐⡀⠅⠠⠁⠂⠄⢀⠠⠐⢀⠂⠨⠀
⢀⢅⠐⡀⠅⢐⠠⣰⠃⠀⠀⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡈⠄⢐⠀⡂⠄⠂⡁⠌⠠⠁⠄⢂⠈⠄⡈⠄⠡
⠃⢨⡂⠄⢂⠐⢠⠇⠀⠀⠈⠐⠑⠁⠀⠀⢀⡀⠄⠀⠀⠀⠀⠀⠀⢸⠐⢈⠠⠐⠀⠄⠡⠀⠌⠠⠁⠌⢀⠂⡁⠄⠨⠐
⢐⢄⠇⢐⠠⠈⡐⡇⡓⢤⡀⠀⠣⡀⠀⢉⠗⡌⠀⠀⠀⠀⠀⠀⠀⠊⣙⠆⡀⢂⢁⠊⢄⠅⢊⠠⢁⠨⢀⠐⡀⠂⡁⠌
⢒⠅⢂⠐⡀⠡⢰⢃⡊⡢⠱⡄⠂⠀⠀⠀⠀⠁⠀⢀⠀⠀⠀⠀⢠⡉⠡⠀⡂⠐⡀⠍⡐⢈⠠⠐⡀⢐⠀⡂⠄⠁⠄⢂
⡊⠌⠠⠐⠠⠈⣏⡢⣊⣌⠪⡘⡆⡀⠀⠀⡈⠒⠂⠁⠕⠅⠀⠀⣸⠠⠈⠄⠂⡁⠄⠂⠄⠂⡐⢀⢂⢐⠠⠐⠀⠅⠨⢀
⠀⢂⠁⠌⡀⠅⠠⠉⡈⢈⠱⢬⡘⢌⠍⢖⢢⡀⠀⠀⠀⠀⠀⣔⠃⠄⠨⢀⠡⠀⢂⠁⢄⡡⣀⡎⠁⢱⢐⠈⠄⠡⢈⠠
⠌⢀⠂⠂⠄⠂⠡⠐⡀⢂⠐⡀⠍⢲⢑⠅⢆⢊⡣⣀⡠⢔⠙⠠⠈⠄⢡⡃⠉⠮⠒⠉⠁⠀⠀⠀⠀⠀⠀⠉⠑⢎⠠⠐
⢂⠐⡀⠅⠂⠡⠈⠄⢐⠀⡂⢐⠀⠹⡔⣅⠵⠋⡈⠄⠂⡐⢈⠠⠁⠌⡠⠑⠀⠀⡀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⢳⡁
⠠⠐⢀⠂⠡⠈⠄⠡⠐⢀⠂⡐⢈⠠⠉⡐⠀⠅⡀⢂⠁⠄⠂⠠⢁⡜⠀⠀⠀⠈⣀⠀⠀⠀⠀⠉⠉⠰⠂⠵⠀⠀⠀⢧
⠂⡈⠄⠄⠡⠈⠄⠡⢈⠠⠐⠠⠐⡠⠕⠚⡆⠁⠄⠂⠄⠡⢈⠐⢰⠁⠀⠀⢀⢀⡁⠀⠠⠜⢒⠁⠀⠀⠁⠀⠀⠀⠀⡳
⢂⠐⠠⠈⠄⠡⠈⠄⠂⡐⠈⠄⡮⠁⠂⣰⢃⠼⠉⢳⠈⢐⠠⠈⠸⡅⠀⠀⠃⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠰⠃
⢂⠈⠄⠡⠈⠄⡁⠂⡁⠄⠨⡸⠐⠀⡼⢁⡞⠀⠀⡎⡂⢐⠠⠈⠄⢳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣏⣀⠼⠐
⡀⠂⠡⢈⡤⠕⠒⠓⠒⠊⠔⠃⡀⡞⢡⡚⠀⢀⠵⢁⠐⠠⠐⢈⠐⡀⠙⠦⠄⠠⠒⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀
⢀⢡⡺⠉⠀⠀⠀⡠⠤⠠⠄⠀⠀⠙⠁⢠⡠⢋⠈⠄⠨⠐⢈⠠⠐⡀⠡⢀⠑⠦⡒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠃⠨
⢠⡓⠀⠀⢀⠀⠂⢀⠤⡀⠀⠀⠄⠀⡂⠀⠒⣄⠌⠠⢁⠈⠄⢐⠀⡂⠨⠀⢂⠁⠱⢄⡀⠀⠀⢀⠀⢠⠀⡖⠋⠄⠂⡁
⢺⠀⠀⠀⠑⠑⠅⠘⠅⠃⠀⠀⠠⠀⠀⢣⠀⠈⣆⠌⠠⠀⠅⢐⠀⡂⠨⠐⠠⢈⠐⠠⠉⡉⢋⠉⠣⠼⠒⡑⢈⠠⠁⠄
⠘⠄⠀⠐⠀⠐⠀⠀⡣⢔⣀⠐⢝⠜⠀⠘⠀⠀⢸⠄⠂⠡⠈⠄⠂⡐⠀⠅⢂⠐⢈⠠⠁⡐⠠⠈⠄⡁⢂⠐⠠⠐⢈⠠
⣎⠀⡆⠀⠄⠀⠄⠈⠌⠁⠀⠀⠀⢔⢄⠄⠀⠀⢸⠕⢈⠠⠁⢂⠁⠄⠡⢈⠠⠐⡀⠂⡁⠄⠂⠡⠐⢀⠂⡈⠐⡈⠠⠐
⡞⠒⠃⠀⠀⠀⡀⠀⠀⠀⡀⠄⠈⠀⠈⠀⠀⢨⡺⠐⢀⠂⡁⢐⠀⠅⠂⡐⢀⠂⠄⠡⠀⡂⠡⢈⠐⡀⠂⢄⠡⢀⢂⠁
⢧⠀⠀⠀⡀⠁⠀⠀⠈⠀⠀⠀⠀⡀⢈⣀⠴⠋⠠⠐⡀⢂⠐⡀⠂⠌⡀⢂⢐⢄⢅⠂⡁⠄⢂⠐⡀⠂⡑⢐⢈⠂⡁⠄
⢈⠇⢠⡀⠀⠀⡀⠐⠀⠐⢰⠒⠂⡸⠁⠡⠀⠅⠂⡁⠄⢀⠂⠄⠁⡂⢐⠀⡇⡃⢎⡢⠂⡐⢀⠂⠄⡁⠄⠂⠄⠂⠄⠌
⠘⠦⠋⡘⠶⢤⠄⢀⠀⢄⣠⠗⠃⠅⠨⠀⠅⠂⡁⠄⢂⠐⠠⢀⠐⡀⠂⠄⠡⠝⠘⡠⢢⠐⡀⠂⡁⠄⠂⡁⠄⡁⢂⠨
⠐⡀⠡⠀⠄⠧⡔⠋⡉⢁⠂⠄⠡⠈⠄⡁⢂⠁⠄⢂⠐⢈⠐⡀⠂⠄⠡⠈⠄⠡⠈⣊⡠⢑⠀⡁⠄⠂⡁⠄⠂⡐⠠⠐
""",
]
