from rq import Worker, Queue, Connection
from django.core.management.base import BaseCommand
from worker.connection import conn, listen


class Command(BaseCommand):
    def handle(self, *args, **options):
        with Connection(conn):
            worker = Worker(map(Queue, listen))
            worker.work()