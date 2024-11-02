import os
import smtplib
import schedule
import time
import csv
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Email configuration
EMAIL_SENDER = "jose@montini.tech"
EMAIL_RECEIVER = "jvd.monteiro@gmail.com"
EMAIL_SUBJECT = "Relatorio Ocorrencias - EPI"
EMAIL_BODY = "Em anexo arquivo com ocorrencias do dia de hoje"
EMAIL_SMTP_SERVER = "smtp.office365.com"
EMAIL_SMTP_PORT = 587
EMAIL_PASSWORD = "xmhgjjdnjhbbgwrg"

# CSV file path
CSV_FILE_PATH = "/opt/nvidia/deepstream/deepstream-7.0/sources/deepstream_python_apps/apps/deepstream-test3-copy/alerts_log.csv"  # Change to the correct path
DAILY_REPORT_DIR = "/opt/nvidia/deepstream/deepstream-7.0/sources/deepstream_python_apps/apps/deepstream-test3-copy/reports"  # Directory to store daily reports


# Function to filter and condense today's entries in the CSV file
def filter_and_condense_csv_for_today():
    """Filter the original CSV to keep only today's entries and condense entries within the same minute."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    daily_report_filename = f"daily_report_{today_str}.csv"
    daily_report_path = os.path.join(DAILY_REPORT_DIR, daily_report_filename)

    # Ensure the daily report directory exists
    os.makedirs(DAILY_REPORT_DIR, exist_ok=True)

    condensed_data = {}

    with open(CSV_FILE_PATH, mode='r') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Assuming first row is header

        for row in reader:
            if len(row) < 4:
                print(f"Skipping row with unexpected format: {row}")
                continue

            camera, date, time, confidence = row[0], row[1], row[2], row[3]

            # Only process today's entries
            if date != today_str:
                continue

            # Combine date and time for minute-level grouping
            timestamp = f"{date} {time[:5]}"  # YYYY-MM-DD HH:MM format
            if timestamp not in condensed_data:
                condensed_data[timestamp] = {
                    "camera": camera,
                    "timestamp": timestamp,
                    "confidences": []
                }

            try:
                # Strip any surrounding whitespace and convert confidence to float
                confidence_float = float(confidence.strip())
                condensed_data[timestamp]["confidences"].append(confidence_float)
            except ValueError:
                print(f"Skipping invalid confidence value: {confidence} in row {row}")
                continue

    # Write condensed data to the daily report CSV
    with open(daily_report_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Camera", "Timestamp", "Average Confidence"])

        for data in condensed_data.values():
            if data["confidences"]:
                avg_confidence = sum(data["confidences"]) / len(data["confidences"])
                writer.writerow([data["camera"], data["timestamp"], f"{avg_confidence:.2f}"])

    print(f"Daily report saved: {daily_report_path}")
    return daily_report_path

# Function to send the condensed CSV file via email
def send_email_with_csv(daily_report_path):
    """Function to send the CSV file via email."""
    try:
        # Set up the email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = EMAIL_SUBJECT

        # Attach the body with the msg instance
        msg.attach(MIMEText(EMAIL_BODY, 'plain'))

        # Open the file to be sent
        with open(daily_report_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(daily_report_path)}")
            msg.attach(part)

        # Create the server connection and send the email
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        print(f"Email sent successfully with {daily_report_path} at 12 PM.")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


# Schedule the task to filter, condense, and email the daily report at 12 PM
def daily_task():
    """Function to schedule the email-sending task at 12 PM daily."""
    print("Scheduling daily email at 12 PM.")
    schedule.every().day.at("10:54").do(send_daily_report)

    while True:
        schedule.run_pending()
        time.sleep(60)


# Function to generate and send the daily report
def send_daily_report():
    """Generate the daily report and send it via email."""
    daily_report_path = filter_and_condense_csv_for_today()
    send_email_with_csv(daily_report_path)


if __name__ == "__main__":
    daily_task()