from rq import Queue
from worker.connection import conn
from pages.views import home_view
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        q = Queue(connection=conn)
        result = q.enqueue(home_view)