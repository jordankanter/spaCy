import pytest

from spacy.strings import StringStore


@pytest.fixture
def stringstore():
    return StringStore()


def test_string_hash(stringstore):
    """Test that string hashing is stable across platforms"""
    assert stringstore.add("apple") == 8566208034543834098
    heart = "\U0001f499"
    h = stringstore.add(heart)
    assert h == 11841826740069053588


def test_stringstore_from_api_docs(stringstore):
    apple_hash = stringstore.add("apple")
    assert apple_hash == 8566208034543834098
    assert stringstore[apple_hash] == "apple"
    assert "apple" in stringstore
    assert "cherry" not in stringstore
    stringstore.add("orange")
    all_strings = [s for s in stringstore]
    assert all_strings == ["apple", "orange"]
    assert all_strings == list(stringstore.keys())
    all_strings_and_hashes = list(stringstore.items())
    assert all_strings_and_hashes == [
        ("apple", 8566208034543834098),
        ("orange", 2208928596161743350),
    ]
    all_hashes = list(stringstore.values())
    assert all_hashes == [8566208034543834098, 2208928596161743350]
    banana_hash = stringstore.add("banana")
    assert len(stringstore) == 3
    assert banana_hash == 2525716904149915114
    assert stringstore[banana_hash] == "banana"
    assert stringstore["banana"] == banana_hash


@pytest.mark.parametrize(
    "val_bytes,val_float,val_list,val_text,val_hash",
    [(b"Hello", 1.1, ["abc"], "apple", 8566208034543834098)],
)
def test_stringstore_type_checking(
    stringstore, val_bytes, val_float, val_list, val_text, val_hash
):
    with pytest.raises(TypeError):
        assert stringstore[val_bytes]

    with pytest.raises(TypeError):
        stringstore.add(val_float)

    with pytest.raises(TypeError):
        assert val_list not in stringstore

    key = stringstore.add(val_text)
    assert val_hash == key
    assert stringstore[val_hash] == val_text


@pytest.mark.parametrize("text1,text2,text3", [("Hello", "goodbye", "hello")])
def test_stringstore_save_unicode(stringstore, text1, text2, text3):
    key = stringstore.add(text1)
    assert stringstore[text1] == key
    assert stringstore[text2] != key
    assert stringstore[text3] != key


@pytest.mark.parametrize("text", ["A"])
def test_stringstore_retrieve_id(stringstore, text):
    key = stringstore.add(text)
    assert len(stringstore) == 1
    assert stringstore[key] == text
    with pytest.raises(KeyError):
        stringstore[20000]


@pytest.mark.parametrize("text1,text2", [("0123456789", "A")])
def test_stringstore_med_string(stringstore, text1, text2):
    store = stringstore.add(text1)
    assert stringstore[store] == text1
    stringstore.add(text2)
    assert stringstore[text1] == store


def test_stringstore_long_string(stringstore):
    text = "INFORMATIVE](http://www.google.com/search?as_q=RedditMonkey&amp;hl=en&amp;num=50&amp;btnG=Google+Search&amp;as_epq=&amp;as_oq=&amp;as_eq=&amp;lr=&amp;as_ft=i&amp;as_filetype=&amp;as_qdr=all&amp;as_nlo=&amp;as_nhi=&amp;as_occt=any&amp;as_dt=i&amp;as_sitesearch=&amp;as_rights=&amp;safe=off"
    store = stringstore.add(text)
    assert stringstore[store] == text


@pytest.mark.parametrize("factor", [254, 255, 256])
def test_stringstore_multiply(stringstore, factor):
    text = "a" * factor
    store = stringstore.add(text)
    assert stringstore[store] == text


def test_stringstore_massive_strings(stringstore):
    text = "a" * 511
    store = stringstore.add(text)
    assert stringstore[store] == text
    text2 = "z" * 512
    store = stringstore.add(text2)
    assert stringstore[store] == text2
    text3 = "1" * 513
    store = stringstore.add(text3)
    assert stringstore[store] == text3


@pytest.mark.parametrize("text", ["qqqqq"])
def test_stringstore_to_bytes(stringstore, text):
    store = stringstore.add(text)
    serialized = stringstore.to_bytes()
    new_stringstore = StringStore().from_bytes(serialized)
    assert new_stringstore[store] == text
