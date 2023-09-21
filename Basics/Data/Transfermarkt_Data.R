# This script gets you onto the player's page
# From there you can access most of the data 
# Selector Gadget can be very helpful in finding the html elements of interest

# get the search page link emulating a search after players name on transfermarkt
player_name = 'Erling Haaland'
player_search_link = paste0("https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=", 
                            gsub(' ', '+', name))
player_search_page = read_html(player_search_link)

# who are the search results
player_found = player_search_page %>% html_nodes("#yw0 .hauptlink a") %>% html_attr('title')

# the links to the player's pages
player_link = player_search_page %>% html_nodes("#yw0 .hauptlink a") %>% html_attr('href') %>% paste0("https://www.transfermarkt.com", .)


# stats page
stats_link = gsub("profil", 'leistungsdatendetails', player_link)
stats_page = read_html(stats_link[1])

minutes_played = app_page %>% html_nodes(".rechts:nth-child(9)") %>% html_children()
