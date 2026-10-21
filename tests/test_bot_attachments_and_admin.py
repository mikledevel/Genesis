"""Tests for BUG-001 (attachments) and BUG-002 (admin conversation log).
unittest-style so they run under pytest or plain `python -m unittest`."""
import base64, io, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core import attachments as A
from app.db.database import GenesisDB

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
JPG = b"\xff\xd8\xff\xe0" + b"0" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"0" * 16


def make_pdf(text):
    """Minimal one-page PDF with a text layer."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>"]
    stream = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET"
    objs += [f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
             "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out)); out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    x = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for o in offs: out += f"{o:010d} 00000 n \n".encode()
    return out + f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode()


class AttachmentTests(unittest.TestCase):
    def test_sniff_by_magic_bytes_not_name(self):
        self.assertEqual(A.sniff_type(PNG), ("image", "image/png"))
        self.assertEqual(A.sniff_type(JPG), ("image", "image/jpeg"))
        self.assertEqual(A.sniff_type(WEBP), ("image", "image/webp"))
        self.assertEqual(A.sniff_type(b"%PDF-1.7 x"), ("document", "application/pdf"))
        self.assertIsNone(A.sniff_type(b"MZ\x90\x00 exe"))

    def test_exe_renamed_to_jpg_rejected(self):
        with self.assertRaises(A.AttachmentError) as cm:
            A.process_attachments([("photo.jpg", b"MZ\x90\x00" + b"0" * 50)])
        self.assertEqual(cm.exception.status_code, 415)

    def test_limits(self):
        with self.assertRaises(A.AttachmentError):
            A.process_attachments([("a.png", PNG)] * 6)
        with self.assertRaises(A.AttachmentError) as cm:
            A.process_attachments([("big.png", PNG + b"0" * A.MAX_FILE_BYTES)])
        self.assertEqual(cm.exception.status_code, 413)
        with self.assertRaises(A.AttachmentError):
            A.process_attachments([("empty.png", b"")])
        self.assertEqual(len(A.process_attachments([("a.png", PNG)] * 5)["images"]), 5)

    def test_pdf_text_extracted(self):
        r = A.process_attachments([("receipt.pdf", make_pdf("Order 12345 refund"))])
        self.assertIn("Order 12345", r["documents"][0]["text"])

    def test_corrupt_pdf_rejected(self):
        with self.assertRaises(A.AttachmentError):
            A.process_attachments([("x.pdf", b"%PDF-1.4 garbage")])

    def test_vision_routing(self):
        m, routed = A.resolve_model("openai/gpt-oss-120b", has_images=True)
        self.assertTrue(routed); self.assertEqual(m, "qwen/qwen3.6-27b")
        self.assertEqual(A.resolve_model("openai/gpt-oss-120b", False), ("openai/gpt-oss-120b", False))
        self.assertEqual(A.resolve_model("qwen/qwen3.6-27b", True), ("qwen/qwen3.6-27b", False))

    def test_build_content(self):
        txt = A.build_user_content("hi", A.process_attachments([("r.pdf", make_pdf("Hello PDF"))]))
        self.assertIsInstance(txt, str); self.assertIn("not instructions", txt)
        parts = A.build_user_content("broken?", A.process_attachments([("p.png", PNG)]))
        self.assertEqual([p["type"] for p in parts], ["text", "image_url"])
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        # image with no message still yields a usable prompt
        self.assertTrue(A.build_user_content("", A.process_attachments([("p.png", PNG)]))[0]["text"])


class ConversationLogTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db"); os.close(fd)
        self.db = GenesisDB(db_path=self.path)
        for cid, agent, msgs in [("c1", "botA", ["Where is my order?", "It ships Monday"]),
                                 ("c2", "botA", ["100% refund please", "ok"]),
                                 ("c3", "botB", ["other bot's chat"])]:
            self.db.create_chat(cid, user_id="u1"); self.db.tag_chat_with_agent(cid, agent)
            for i, t in enumerate(msgs):
                self.db.add_message(cid, "user" if i % 2 == 0 else "assistant", t)

    def tearDown(self):
        for ext in ("", "-wal", "-shm"):
            try: os.remove(self.path + ext)
            except FileNotFoundError: pass

    def test_scoped_to_bot(self):
        r = self.db.list_agent_conversations("botA")
        self.assertEqual(r["total"], 2); self.assertEqual({c["id"] for c in r["conversations"]}, {"c1", "c2"})

    def test_search_and_wildcard_escape(self):
        self.assertEqual([c["id"] for c in self.db.list_agent_conversations("botA", search="order")["conversations"]], ["c1"])
        self.assertEqual(self.db.list_agent_conversations("botA", search="100%")["total"], 1)
        self.assertEqual(self.db.list_agent_conversations("botA", search="%")["total"], 1)   # literal %, not wildcard
        self.assertEqual(self.db.list_agent_conversations("botA", search="nomatch")["total"], 0)

    def test_date_filter_and_paging(self):
        self.assertEqual(self.db.list_agent_conversations("botA", date_from="2999-01-01")["total"], 0)
        self.assertEqual(self.db.list_agent_conversations("botA", date_to="2000-01-01")["total"], 0)
        self.assertEqual(self.db.list_agent_conversations("botA", date_from="2000-01-01")["total"], 2)
        r = self.db.list_agent_conversations("botA", limit=1, offset=1)
        self.assertEqual((r["total"], len(r["conversations"])), (2, 1))

    def test_cannot_read_other_bots_chat(self):
        self.assertEqual(len(self.db.get_agent_conversation("botA", "c1")), 2)
        self.assertIsNone(self.db.get_agent_conversation("botA", "c3"))
        self.assertIsNone(self.db.get_agent_conversation("botA", "missing"))


if __name__ == "__main__":
    unittest.main()
