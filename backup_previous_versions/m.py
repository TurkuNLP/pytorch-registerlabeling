from google.oauth2.credentials import Credentials
import base64
from email.message import EmailMessage

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = "https://www.googleapis.com/auth/gmail.send"
CLIENT_SECRET_FILE = "credentials.json"
APPLICATION_NAME = "Gmail API Python Send Email"


def get_credentials():
    return Credentials.from_authorized_user_file("token.json", SCOPES)


def main():
    """Create and send an email message
    Print the returned  message id
    Returns: Message object, including message id

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    creds = get_credentials()

    try:
        service = build("gmail", "v1", credentials=creds)
        message = EmailMessage()

        message.set_content("This is automated draft mail")

        message["To"] = "pytorchregisterlabeling@gmail.com"
        message["From"] = "pytorchregisterlabeling@gmail.com"
        message["Subject"] = "Automated draft"

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
        # pylint: disable=E1101
        send_message = (
            service.users().messages().send(userId="me", body=create_message).execute()
        )
        print(f'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(f"An error occurred: {error}")
        send_message = None
    return send_message


if __name__ == "__main__":
    main()
