"""SMTP email notifier — single-message, digest, and report-attachment senders."""

from __future__ import annotations

import asyncio
import html
import smtplib
from datetime import datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from application.services.notification.types import EmailConfig, NotificationMessage


class EmailNotifier:
    def __init__(self, cfg: EmailConfig):
        self.cfg = cfg

    async def send(self, msg: NotificationMessage, to_emails: Optional[List[str]] = None) -> None:
        if not self.cfg.enabled:
            return

        recipients = (to_emails or []) or self.cfg.to_emails
        if not recipients:
            return

        await asyncio.to_thread(self._send_sync, msg, recipients)

    async def send_digest(
        self,
        messages: List[NotificationMessage],
        to_emails: Optional[List[str]] = None,
    ) -> None:
        ordered = sorted(messages or [], key=lambda msg: int(msg.ts_ms), reverse=True)
        if not ordered:
            return
        if len(ordered) == 1:
            await self.send(ordered[0], to_emails=to_emails)
            return

        if not self.cfg.enabled:
            return

        recipients = (to_emails or []) or self.cfg.to_emails
        if not recipients:
            return

        await asyncio.to_thread(self._send_digest_sync, ordered, recipients)

    async def send_report(
        self,
        *,
        subject: str,
        html_body: str,
        text_body: str,
        attachment_bytes: bytes,
        attachment_filename: str,
        to_emails: List[str],
        attachment_subtype: str = "pdf",
    ) -> bool:
        """Send a one-off email with a binary attachment (e.g. a PDF report).

        Returns True when the message was handed to SMTP, False when email is
        disabled or there are no recipients. Runs the blocking SMTP exchange on
        a worker thread so it never stalls the event loop.
        """
        if not self.cfg.enabled:
            return False
        recipients = [e for e in (to_emails or []) if e]
        if not recipients:
            return False

        await asyncio.to_thread(
            self._send_report_sync,
            subject,
            html_body,
            text_body,
            attachment_bytes,
            attachment_filename,
            recipients,
            attachment_subtype,
        )
        return True

    def _smtp_send(self, message: MIMEMultipart, to_emails: List[str], *, timeout: int = 10) -> None:
        """Connect, optionally STARTTLS + login, and send one MIME message.

        Single SMTP exchange shared by every sender (alert, digest, report) so
        the connect/login/sendmail dance lives in exactly one place.
        """
        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=timeout) as server:
            if self.cfg.use_tls:
                server.starttls()
            if self.cfg.smtp_user and self.cfg.smtp_pass:
                server.login(self.cfg.smtp_user, self.cfg.smtp_pass)
            server.sendmail(self.cfg.from_email, to_emails, message.as_string())

    def _send_report_sync(
        self,
        subject: str,
        html_body: str,
        text_body: str,
        attachment_bytes: bytes,
        attachment_filename: str,
        to_emails: List[str],
        attachment_subtype: str,
    ) -> None:
        outer = MIMEMultipart("mixed")
        outer["Subject"] = subject
        outer["From"] = self.cfg.from_email
        outer["To"] = ", ".join(to_emails)

        body = MIMEMultipart("alternative")
        body.attach(MIMEText(text_body, "plain", "utf-8"))
        body.attach(MIMEText(html_body, "html", "utf-8"))
        outer.attach(body)

        part = MIMEApplication(attachment_bytes, _subtype=attachment_subtype)
        part.add_header(
            "Content-Disposition", "attachment", filename=attachment_filename
        )
        outer.attach(part)

        self._smtp_send(outer, to_emails, timeout=20)

    def _send_sync(self, msg: NotificationMessage, to_emails: List[str]) -> None:
        subject = f"{self.cfg.subject_prefix} {msg.site_name} — {msg.title}"

        text_body = self._render_text(msg)
        html_body = self._render_html(msg)

        m = MIMEMultipart("alternative")
        m["Subject"] = subject
        m["From"] = self.cfg.from_email
        m["To"] = ", ".join(to_emails)

        m.attach(MIMEText(text_body, "plain", "utf-8"))
        m.attach(MIMEText(html_body, "html", "utf-8"))

        self._smtp_send(m, to_emails)

    def _send_digest_sync(self, messages: List[NotificationMessage], to_emails: List[str]) -> None:
        first = messages[0]
        total = len(messages)
        subject = f"{self.cfg.subject_prefix} {first.site_name} — {total} alerts"

        text_body = self._render_digest_text(messages)
        html_body = self._render_digest_html(messages)

        m = MIMEMultipart("alternative")
        m["Subject"] = subject
        m["From"] = self.cfg.from_email
        m["To"] = ", ".join(to_emails)

        m.attach(MIMEText(text_body, "plain", "utf-8"))
        m.attach(MIMEText(html_body, "html", "utf-8"))

        self._smtp_send(m, to_emails)

    def _render_text(self, msg: NotificationMessage) -> str:
        ts = self._fmt_ts(msg.ts_ms)
        classes = ", ".join(msg.cls_names) if msg.cls_names else "unknown"
        conf = f"{msg.max_conf:.2f}" if msg.max_conf is not None else "n/a"

        lines = [
            "1886NOENTRY — AI Monitor Alert",
            "",
            f"Site: {msg.site_name}",
            f"Camera: {msg.camera_name or msg.camera_uuid}",
            f"Time (UTC): {ts}",
            f"Detected: {classes}",
            f"Max confidence: {conf}",
            "",
            msg.body,
        ]

        if self.cfg.dashboard_base_url:
            url = self._camera_url(msg.camera_uuid)
            lines += ["", f"Open dashboard: {url}"]

        lines += [
            "",
            "If you did not expect this email, check your notification settings in the 1886NOENTRY dashboard.",
        ]
        return "\n".join(lines)

    def _render_html(self, msg: NotificationMessage) -> str:
        site = html.escape(msg.site_name or "Unknown Site")
        title = html.escape(msg.title or "Alert")
        body = html.escape(msg.body or "")
        cam = html.escape(msg.camera_name or msg.camera_uuid)
        dev = html.escape(msg.device_name or "")
        ts = html.escape(self._fmt_ts(msg.ts_ms))
        classes = ", ".join(html.escape(c) for c in (msg.cls_names or ["unknown"]))
        conf = f"{msg.max_conf:.2f}" if msg.max_conf is not None else "n/a"

        button_html = ""
        if self.cfg.dashboard_base_url:
            url = html.escape(self._camera_url(msg.camera_uuid))
            button_html = f"""
              <tr>
                <td style="padding: 14px 24px 24px 24px;">
                  <a href="{url}"
                     style="display:inline-block;text-decoration:none;background:#0ea5e9;color:#ffffff;
                            padding:12px 16px;border-radius:10px;font-weight:700;font-family:Arial,sans-serif;">
                    Open Dashboard
                  </a>
                </td>
              </tr>
            """

        return f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#f4f6fb;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f4f6fb;padding:24px 0;">
      <tr>
        <td align="center">
          <table role="presentation" width="640" cellspacing="0" cellpadding="0"
                 style="background:#ffffff;border-radius:16px;overflow:hidden;border:1px solid #e6e8ef;">
            <tr>
              <td style="padding:18px 24px;background:#0b1220;color:#ffffff;font-family:Arial,sans-serif;">
                <div style="font-size:14px;opacity:0.85;font-weight:700;letter-spacing:0.3px;">1886NOENTRY</div>
                <div style="font-size:22px;font-weight:800;margin-top:6px;">AI Monitor Alert</div>
              </td>
            </tr>

            <tr>
              <td style="padding:18px 24px;font-family:Arial,sans-serif;">
                <div style="display:inline-block;background:#fee2e2;color:#991b1b;padding:6px 10px;border-radius:999px;
                            font-weight:700;font-size:12px;">
                  ALERT
                </div>

                <div style="margin-top:12px;font-size:18px;font-weight:800;color:#111827;">
                  {title}
                </div>

                <div style="margin-top:6px;color:#475569;font-size:14px;line-height:1.5;">
                  A detection event was triggered at <strong>{site}</strong>.
                </div>

                <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                       style="margin-top:14px;border-collapse:separate;border-spacing:0;background:#f8fafc;
                              border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Site</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{site}</td>
                  </tr>
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Camera</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{cam}</td>
                  </tr>
                  {f'''
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Device</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{dev}</td>
                  </tr>
                  ''' if dev else ''}
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Detected</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{classes}</td>
                  </tr>
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Max Confidence</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{conf}</td>
                  </tr>
                  <tr>
                    <td style="padding:12px 14px;color:#0f172a;font-weight:700;font-size:13px;">Time (UTC)</td>
                    <td style="padding:12px 14px;color:#334155;font-size:13px;">{ts}</td>
                  </tr>
                </table>

                <div style="margin-top:14px;color:#0f172a;font-size:14px;line-height:1.6;">
                  {body}
                </div>
              </td>
            </tr>

            {button_html}

            <tr>
              <td style="padding:14px 24px 18px 24px;background:#f8fafc;border-top:1px solid #e2e8f0;
                         font-family:Arial,sans-serif;color:#64748b;font-size:12px;line-height:1.5;">
                You are receiving this alert because notifications are enabled for this site/camera in 1886NOENTRY.
                If this is unexpected, adjust notification settings in your dashboard.
              </td>
            </tr>

          </table>

          <div style="width:640px;text-align:center;color:#94a3b8;font-family:Arial,sans-serif;font-size:12px;margin-top:10px;">
            © {datetime.now().year} 1886NOENTRY
          </div>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    def _render_digest_text(self, messages: List[NotificationMessage]) -> str:
        first = messages[0]
        shown = messages[:20]
        remaining = max(0, len(messages) - len(shown))

        lines = [
            "1886NOENTRY — Buffered Alert Digest",
            "",
            f"Site: {first.site_name}",
            f"Alerts in batch: {len(messages)}",
            "",
        ]

        for msg in shown:
            lines.append(
                f"- [{self._fmt_ts(msg.ts_ms)} UTC] {msg.title}: {msg.body}"
            )

        if remaining > 0:
            lines += ["", f"... and {remaining} more alerts in this batch."]

        if self.cfg.dashboard_base_url:
            lines += ["", f"Open dashboard: {self._camera_url(first.camera_uuid)}"]

        return "\n".join(lines)

    def _render_digest_html(self, messages: List[NotificationMessage]) -> str:
        first = messages[0]
        shown = messages[:20]
        remaining = max(0, len(messages) - len(shown))

        rows = []
        for msg in shown:
            rows.append(
                f"""
                <tr>
                  <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-size:13px;white-space:nowrap;">{html.escape(self._fmt_ts(msg.ts_ms))}</td>
                  <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-size:13px;font-weight:700;">{html.escape(msg.title or "Alert")}</td>
                  <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{html.escape(msg.body or "")}</td>
                </tr>
                """
            )

        more_html = ""
        if remaining > 0:
            more_html = f"""
              <div style="margin-top:12px;color:#64748b;font-size:13px;">
                ... and {remaining} more alerts in this batch.
              </div>
            """

        button_html = ""
        if self.cfg.dashboard_base_url:
            url = html.escape(self._camera_url(first.camera_uuid))
            button_html = f"""
              <div style="margin-top:16px;">
                <a href="{url}"
                   style="display:inline-block;text-decoration:none;background:#0ea5e9;color:#ffffff;
                          padding:12px 16px;border-radius:10px;font-weight:700;font-family:Arial,sans-serif;">
                  Open Dashboard
                </a>
              </div>
            """

        return f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:24px;background:#f4f6fb;font-family:Arial,sans-serif;">
    <div style="max-width:760px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;">
      <div style="padding:18px 24px;background:#0b1220;color:#ffffff;">
        <div style="font-size:14px;opacity:0.85;font-weight:700;">1886NOENTRY</div>
        <div style="font-size:22px;font-weight:800;margin-top:6px;">Buffered Alert Digest</div>
      </div>
      <div style="padding:18px 24px;">
        <div style="color:#0f172a;font-size:16px;font-weight:800;">{html.escape(first.site_name or "Unknown Site")}</div>
        <div style="margin-top:6px;color:#475569;font-size:14px;">{len(messages)} alerts were buffered and delivered together.</div>
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
               style="margin-top:16px;border-collapse:separate;border-spacing:0;background:#f8fafc;
                      border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
          <thead>
            <tr>
              <th style="padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;color:#0f172a;font-size:12px;">Time (UTC)</th>
              <th style="padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;color:#0f172a;font-size:12px;">Title</th>
              <th style="padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;color:#0f172a;font-size:12px;">Details</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
        {more_html}
        {button_html}
      </div>
    </div>
  </body>
</html>
"""

    def _fmt_ts(self, ts_ms: int) -> str:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _camera_url(self, camera_uuid: str) -> str:
        base = (self.cfg.dashboard_base_url or "").rstrip("/")
        path = self.cfg.camera_path_template.format(camera_uuid=camera_uuid)
        return f"{base}{path}"
