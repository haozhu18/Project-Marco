from rq import Queue
from oursite.worker.connection import conn
from oursite.pages.views import home_view
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        q = Queue(connection=conn)
        result = q.enqueue(home_view)