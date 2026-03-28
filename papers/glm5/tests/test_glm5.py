from glm5 import main


def test_main(capsys):
    main()
    assert "glm5 ready" in capsys.readouterr().out
