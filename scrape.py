from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from bs4 import BeautifulSoup
import time
import pandas as pd

driver = webdriver.Chrome()

def extract_page_data(driver):
    results = []
    page_source = driver.page_source
    
    soup = BeautifulSoup(page_source, 'html.parser')
    res = soup.find_all("div", class_= "docsum-content")
    for result in res:
        title = result.find("a", class_="docsum-title")
        authors = result.find("span", class_="docsum-authors full-authors")
        abstract = result.find("div", class_="full-view-snippet")
        pmid = result.find("span", class_="docsum-pmid")
        citation = result.find("span", class_ = "docsum-journal-citation short-journal-citation")

        if title and authors:
            title_text = title.text.strip()
            authors_text = authors.text.strip()
            abstract_text = abstract.text.strip() if abstract else "Abstract not available"
            pmid = pmid.text.strip()
            journal = citation.text.strip()
            year, journal_ = int(journal.split(" ")[-1].replace('.', '')), " ".join(journal.split(" ")[:-1])
            
            results.append({
                "Title": title_text,
                "Authors": authors_text,
                "Abstract": abstract_text,
                "PMID": pmid,
                "Journal": journal_,
                "Year":year
            })
    return results

if __name__ == "__main__":
    final_results = []
    search_term = ["Network science", "cryptography", "quantum physics", "neural network", "machine learning", "information retrieval"]
    driver.get('https://pubmed.ncbi.nlm.nih.gov/')

    for term in search_term:
        search_input = driver.find_element(By.ID, "id_term")
        search_input.send_keys(Keys.CONTROL + "a")  # Select all text in the input field
        search_input.send_keys(Keys.DELETE)  
        search_input.send_keys(term)
        driver.find_element(By.CLASS_NAME, "search-btn").click()
        for i in range(2):
            page_output = extract_page_data(driver)
            final_results.extend(page_output)
            
            next_page = driver.find_element(By.CLASS_NAME, "next-page-btn")
            if "disabled" in next_page.get_attribute("class"):
                break
            
            next_page.click()
            time.sleep(2)
    df = pd.DataFrame(final_results)
    df = df.dropna(subset=['abstract'])
    df.to_csv("F://IR/Data/Data.csv")
    driver.quit()