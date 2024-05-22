from ui import app

app.queue(status_update_rate=10.0, max_size=10).launch()
