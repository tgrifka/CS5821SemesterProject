import requests
import pandas as pd
import time

def download_image(filename,url):
    r = requests.get(url, allow_redirects=True)
    temp = open(filename, 'wb')
    temp.write(r.content)

def colorListTocolorString(colorList):
    string = ""
    for ele in colorList:
        if ele != "[" and ele != "]" and ele != ",":
            string += str(ele)
    if string == "":
        return "None"
    return string

def generateFileName(id, colorString):
    filename = "Images/" + colorString + "/" + id + ".png"
    return filename

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_json("../../unique-artwork-20221004090238.json")
    data["colorString"] = data["color_identity"].apply(lambda colorList: colorListTocolorString(colorList))
    data.drop(axis=1, columns=['object','name', 'oracle_id', 'multiverse_ids', 'mtgo_id',
       'mtgo_foil_id', 'tcgplayer_id', 'cardmarket_id', 'lang',
       'released_at', 'uri', 'scryfall_uri', 'layout', 'highres_image',
       'image_status', 'mana_cost', 'cmc', 'type_line',
       'oracle_text', 'power', 'toughness', 'colors', 'color_identity',
       'keywords', 'legalities', 'games', 'reserved', 'foil', 'nonfoil',
       'finishes', 'oversized', 'promo', 'reprint', 'variation', 'set_id',
       'set', 'set_name', 'set_type', 'set_uri', 'set_search_uri',
       'scryfall_set_uri', 'rulings_uri', 'prints_search_uri',
       'collector_number', 'digital', 'rarity', 'flavor_text', 'card_back_id',
       'artist', 'artist_ids', 'illustration_id', 'border_color', 'frame',
       'full_art', 'textless', 'booster', 'story_spotlight', 'edhrec_rank',
       'penny_rank', 'prices', 'related_uris', 'arena_id', 'preview',
       'security_stamp', 'promo_types', 'produced_mana', 'watermark',
       'all_parts', 'loyalty', 'frame_effects', 'card_faces',
       'attraction_lights', 'color_indicator', 'life_modifier',
       'hand_modifier', 'tcgplayer_etched_id', 'content_warning',
       'printed_name', 'printed_type_line', 'printed_text', 'flavor_name',
       'variation_of'], inplace=True)
    data.dropna(inplace=True)
    count = 0
    for row in data.to_numpy():
        print(row[0] + " " +row[1]["art_crop"] + " " + row[2])
        fn = generateFileName(row[0], row[2])
        download_image(fn, row[1]["art_crop"])
        count += 1
        time.sleep(0.2)

    print(count)