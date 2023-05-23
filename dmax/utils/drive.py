from __future__ import print_function
import requests
import pickle
import os
import io
import os.path
from tqdm import tqdm

from googleapiclient.discovery import build, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials

# If modifying these scopes, delete the file token.pickle.
SCOPES = {
    "rw": ['https://www.googleapis.com/auth/drive'],
    "ro": ['https://www.googleapis.com/auth/drive.readonly']
}


def build_service(mode="ro"):
    """
    taken from https://developers.google.com/drive/api/v3/quickstart/python
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is created automatically when
    # the authorization flow completes for the first time.
    if os.path.exists('../token.pickle'):
        with open('../token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if os.path.exists('../token.json'):
        with open('../token.json', 'rb') as token:
            creds = Credentials.from_authorized_user_file('../token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('../credentials.json'):
                raise FileNotFoundError(
                    "`credentials.json` not found. Make sure to rename the downloaded JSON file `client_secret[...]json` to `credentials.json`"
                    " and to run python `data/gdrive.py init` under the dot-dmax/dmax directory, on a computed with a display."
                )
            flow = InstalledAppFlow.from_client_secrets_file('../credentials.json', SCOPES[mode])
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('../token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service


def upload_folder_helper(service, folder_name, parent_path, parent_id=None, recurse=True, verbose=True):
    """
    Uploads the given folder and its content to the provided service
    Return IDs, a dictionary containing all the ids of the uploaded files.
    """
    ids = {}
    if parent_id is not None:
        file_metadata = {
            'name': folder_name,
            'parents': [parent_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
    file = service.files().create(body=file_metadata, fields='id').execute()
    folder_id = file.get('id')
    for file in os.listdir(os.path.join(parent_path, folder_name)):
        if os.path.isdir(os.path.join(parent_path, folder_name, file)):
            if recurse:
                ids[file] = upload_folder_helper(service, file, os.path.join(parent_path, folder_name),
                                                 parent_id=folder_id, recurse=recurse, verbose=verbose)
            else:
                continue
        else:
            file_metadata = {
                'name': file,
                'parents': [folder_id]
            }
            file_path = os.path.join(parent_path, folder_name, file)
            media = MediaFileUpload(file_path, resumable=True)
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            ids[file_path] = file.get('id')
            if verbose:
                print(f"uploaded {file_path}, id = {ids[file_path]}")
    return ids


def upload_folder(service, folder_path, recurse=True, verbose=True):
    return upload_folder_helper(
        service,
        os.path.basename(folder_path),
        os.path.dirname(folder_path),
        parent_id=None,
        recurse=recurse,
        verbose=verbose
    )


def download_file(service, path_to_save, file_id, verbose=True):
    if verbose:
        print(f"downloading {path_to_save}, id = {file_id}")

    if service is None:
        r = requests.get(file_id, allow_redirects=True)
        with open(path_to_save, "wb") as f:
            f.write(r.content)
        return

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    pbar = None
    if verbose:
        pbar = tqdm(total=100)
    while done is False:
        status, done = downloader.next_chunk()
        if verbose:
            pbar.update(int(status.progress() * 100))
    if verbose:
        pbar.close()

    with open(path_to_save, "wb") as f:
        f.write(fh.getbuffer())
