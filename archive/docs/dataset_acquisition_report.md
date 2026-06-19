# UA-DETRAC Dataset Acquisition Report

## Discovery Results

An exhaustive web search across the University at Albany domains, web archives, and major dataset mirrors (Kaggle, Roboflow, GitHub) yielded the following access points for the UA-DETRAC dataset (`DETRAC-train-data.zip` and `DETRAC-test-data.zip`). 

HTTP HEAD requests (`curl -I`) were executed against the official domain and web archives, resulting in indefinite hangs/timeouts, confirming the primary host is offline.

## Acquisition Endpoints

| URL | HOST | STATUS | SIZE | PUBLIC_ACCESS | LIKELIHOOD_VALID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip` | RIT Albany | DEAD (Timeout) | Unknown | NO | HIGH (if online) |
| `http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip` | RIT Albany | DEAD (Timeout) | Unknown | NO | HIGH (if online) |
| `http://web.archive.org/web/20210515000000/http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip` | Archive.org | DEAD (Timeout) | Unknown | NO | LOW |
| `https://www.kaggle.com/datasets?search=UA-DETRAC` | Kaggle | ALIVE | ~10-20GB+ | LOGIN REQUIRED | HIGH (Community Uploads) |
| `https://universe.roboflow.com/search?q=ua-detrac` | Roboflow | ALIVE | Varies | LOGIN REQUIRED | MEDIUM (Often heavily sampled/formatted for YOLO) |

## Acquisition Path Ranking

Ranked from easiest/most viable to hardest/impossible:

### 1. Kaggle Hub (Most Viable)
*   **Path:** Utilize the `kaggle` Python API to download a community-uploaded variant of UA-DETRAC.
*   **Blocker:** Requires the user to authenticate by dropping a valid `kaggle.json` credentials file into `~/.kaggle/`.
*   **Actionable:** Highly automatable via bash/python once credentials are in place.

### 2. Roboflow Universe (Fallback)
*   **Path:** Use the `roboflow` pip package to pull the dataset.
*   **Blocker:** Requires a Roboflow API key. Data is frequently mutilated into bounding-box specific formats (YOLOv8, Pascal VOC) and frame-sampled (e.g., "UA-DETRAC-10K-SAMPLE"), violating the strict 16-frame temporal sequence requirements of VideoMAE unless a raw format is found.

### 3. Academic Contact (Manual)
*   **Path:** Email the original authors (Longyin Wen, Dawei Du) for a private Google Drive or Baidu Pan mirror link.
*   **Blocker:** Strictly manual, asynchronous, and requires academic affiliation proof.

### 4. Official RIT Server (Dead)
*   **Path:** Direct `wget` against the albany.edu subdomains.
*   **Blocker:** The server `detrac-db.rit.albany.edu` is physically unresponsive and dropping all packets. Automated downloads are impossible.
