from ui import app

app.queue(status_update_rate=5.0, max_size=10).launch()
