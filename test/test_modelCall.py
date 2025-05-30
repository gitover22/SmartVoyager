import os
import types
import pytest

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import modelCall as omc  
OpenaiModelsCall = omc.OpenaiModelsCall

#########################
# ---- Fake 对象集 ---- #
#########################
class _FakeChatCompletion:
    def __init__(self, content):
        msg = types.SimpleNamespace(content=content)
        self.choices = [types.SimpleNamespace(message=msg)]

class _FakeChat:
    def __init__(self, content="assistant"):
        self.completions = types.SimpleNamespace(
            create=lambda **_: _FakeChatCompletion(content)
        )

class _FakeOpenAI:
    def __init__(self, content="assistant"):
        self.chat = _FakeChat(content)
         # ➜ 补一个 images.generate
        self.images = types.SimpleNamespace(
            generate=lambda **_: types.SimpleNamespace(
                data=[types.SimpleNamespace(url="http://fake/img")]
            )
        )


def _patch_openai(monkeypatch):
    """统一伪造 openai / requests"""
    # 1) client.OpenAI()
    monkeypatch.setattr(omc, "OpenAI", _FakeOpenAI)

    # 2) 高层 openai.* API
    fake_openai = types.SimpleNamespace(
        Audio=types.SimpleNamespace(
            transcribe=lambda *_: {"text": "transcribed text"}
        ),
        audio=types.SimpleNamespace(
            speech=types.SimpleNamespace(
                create=lambda **_: types.SimpleNamespace(content=b"fake_mp3")
            )
        ),
        Image=types.SimpleNamespace(
            create=lambda **_: {"data": [{"url": "http://fake/img"}]}
        ),
        Embedding=types.SimpleNamespace(
            create=lambda **_: {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        ),
    )
    monkeypatch.setattr(omc, "openai", fake_openai)

    # 3) requests.get → 返回伪二进制
    import requests
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_: types.SimpleNamespace(content=b"fake_image_bytes"),
    )


#############################
# ---- pytest fixture ---- #
#############################
@pytest.fixture
def model(monkeypatch, tmp_path):
    _patch_openai(monkeypatch)

    # 把类级目录指向临时路径
    audio_dir = tmp_path / "audios"
    image_dir = tmp_path / "images"
    audio_dir.mkdir()
    image_dir.mkdir()
    monkeypatch.setattr(OpenaiModelsCall, "temp_audio_dir", str(audio_dir))
    monkeypatch.setattr(OpenaiModelsCall, "temp_image_dir", str(image_dir))

    return OpenaiModelsCall()


####################################
# --------- 单元测试区域 ---------- #
####################################
def test_chat_with_openai(model):
    out = model.chat_with_openai("你好呀")
    assert out == "assistant"


def test_image_understanding(model, tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"png")
    out = OpenaiModelsCall.image_understanding("描述图片", str(img))
    assert out == "assistant"


def test_speech_to_text(model, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    assert model.speech_to_text(str(wav)) == "transcribed text"


def test_generate_image(model):
    path = model.generate_image("宇宙中的猫")
    assert os.path.isfile(path) # 目的是
    with open(path, "rb") as f:
        assert f.read() == b"fake_image_bytes"


def test_text_to_speech(model):
    path = model.text_to_speech("hello", filename="t.mp3")
    assert os.path.isfile(path)
    with open(path, "rb") as f:
        assert f.read() == b"fake_mp3"


def test_text_to_audio(model):
    path = model.text_to_audio("文字转语音")
    assert os.path.isfile(path)


def test_get_openai_embedding(monkeypatch):
    _patch_openai(monkeypatch)
    assert OpenaiModelsCall.get_openai_embedding("文本") == [0.1, 0.2, 0.3]
