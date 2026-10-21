import tasks
from sentiment.collect import Source

ALLOW_ALL = "User-agent: *\nAllow: /\n"


def test_fixtures_record_robots_once_per_host_and_never_content(gate, web, clock, tmp_path):
    robots = web.get("https://a.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://b.test/robots.txt").respond(200, text=ALLOW_ALL)
    sources = [
        Source(name="b", lang="en", rss="https://b.test/rss", frontpage=False),
        Source(name="a", lang="en", rss="https://a.test/rss", frontpage_url="https://a.test/"),
    ]
    tasks.record_fixtures(gate, sources, tmp_path)
    assert robots.call_count == 1
    assert (tmp_path / "robots" / "a.test.txt").read_text() == ALLOW_ALL
    assert sorted(p.name for p in tmp_path.iterdir()) == ["robots"]  # no feeds, no front pages
    assert all(x >= 5.0 for x in clock.slept)


class Recorder:
    def run(self, cmd, env=None, pty=False):
        self.env = env


def test_local_collect_scores_with_the_models_inv_train_wrote(monkeypatch):
    monkeypatch.delenv("SENTIMENT_MODEL_DIR", raising=False)
    c = Recorder()
    tasks.collect.body(c, local=True)
    assert c.env["SENTIMENT_MODEL_DIR"] == str(tasks.ROOT / "models")
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", "/elsewhere")
    tasks.collect.body(c, local=True)
    assert c.env["SENTIMENT_MODEL_DIR"] == "/elsewhere"
