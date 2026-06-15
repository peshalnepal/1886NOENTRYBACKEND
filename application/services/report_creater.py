from application.repositories.notification_repository import NotificationRepository
from application.repositories.site_repository import SiteRepository
class AlertReport():
    def __init__(self, *args, **kwds):
        site_repo=SiteRepository()
        notification_repo=NotificationRepository()
    def create_report(self,site_uuid):
        notificaitions_by_time=None
    def publish_report(self,report):
        pass
    def save_report(self,report):
        pass
    def __call__(self, *args, **kwds):
        pass
    