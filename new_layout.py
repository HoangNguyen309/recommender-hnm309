import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from recommender import recommender
from recommender_user import recommender_user

st.set_page_config(page_title='Hệ thống đề xuất sản phẩm')

st.image("image/csc_banner.png")
st.markdown("<h1 style='text-align: center;'>Đồ án tốt nghiệp Data Science</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Chủ đề: Recommendation System (Tiki.vn)</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Nhóm<br>Nguyễn Minh Hoàng - Trần Trọng Huy</h3>", unsafe_allow_html=True)

products = pd.read_csv("final_product.csv",skipinitialspace=True)
products = products.drop(columns=["Unnamed: 0"]).set_index("index")

reviews = pd.read_csv("final_review.csv",skipinitialspace=True)
reviews = reviews.drop(columns=["Unnamed: 0"]).set_index("id")

dictionary = pickle.load(open('Dictionary.sav', 'rb'))
tfidf = pickle.load(open('TfidfModel.sav', 'rb'))
index = pickle.load(open('Index.sav', 'rb'))

box = st.selectbox("Mục lục:",("Lời mở đầu","Xây dựng hệ thống","Đề xuất sản phẩm khi khách hàng chọn một sản phẩm bất kỳ","Đề xuất sản phẩm bằng ID khách hàng"))
if box == "Lời mở đầu":
    st.image('image/RecommendationEngine-1200x675.png')
    st.write("""
    Mục tiêu xây dựng hệ thống đề xuất sản phẩm 
    """)
    

elif box == "Xây dựng hệ thống":
    st.image("image/toptal-blog-image.png")

    st.write("""
    #### Some data of products:
    """)
    st.dataframe(products.head())
    st.write("""
    #### Some data of reviews:
    """)
    st.dataframe(reviews.head())

    st.write("""
    #### Explode more data:
    """)
    st.dataframe(products[['price','list_price','rating']].describe())
    st.markdown("""
    * 'price' and 'list price' got large range of value trong 7 thousands to ~62.7 milion
    * 'rating' in range 0-5
    """)
    # Visualize the result
    rating = products['rating'].unique().tolist()
    price = products['price'].unique().tolist()
    rating_selection = st.slider('Rating:',
                                min_value= min(rating),
                                max_value= max(rating),
                                value=(min(rating),max(rating)))
    price_selection = st.slider("Price:",
                                min_value= min(price),
                                max_value= max(price),
                                value=(min(price),max(price)))
    # Filter dataframe based on selection rating
    mask = products['rating'].between(*rating_selection)
    number_of_result = products[mask].shape[0]
    st.markdown(f'*Available results: {number_of_result}*')

    brands = products.groupby('brand')['item_id'].count().sort_values(ascending=False)

    plt.subplots_adjust(top=1,bottom=0)
    brands[1:11].plot(kind='bar')
    plt.ylabel('count')
    plt.title('Product Items by brands')
    st.pyplot(plt)

    st.markdown("""
    => Samsung is the brand got a highest number of products.
    """)

    plt.subplots_adjust(top=1,bottom=0)
    price_by_brand = products.groupby(by='brand').mean()['price']
    price_by_brand.sort_values(ascending=False)[:10].plot(kind='bar')
    plt.ylabel('price')
    plt.title('Average price by brand')
    st.pyplot(plt)
    
    st.markdown("""
    => Hitachi has high average price.
    """)

    plt.subplots_adjust(top=1,bottom=0)
    sns.displot(products,x='rating',kind='hist')
    plt.title('Total ratings')
    st.pyplot(plt)

    st.markdown("""
    => Almost rating is 0 and 5 and above 4
    """)

    avg_rating_customer = reviews.groupby(by='product_id').mean()['rating'].to_frame().reset_index()
    avg_rating_customer.rename({'rating':'avg_rating'},axis=1,inplace=True)
    new_products = products.merge(avg_rating_customer,left_on='item_id',right_on='product_id',how='left')


    plt.subplots_adjust(top=1,bottom=0)
    sns.displot(new_products,x='avg_rating',kind='hist')
    plt.title('Average ratings')
    st.pyplot(plt)

    st.markdown("""
    => Rating product by customer > 0. Then we can see rating = 0 in product is missing value.
    """)


    plt.subplots_adjust(top=1,bottom=0)
    sns.displot(reviews,x='rating',kind='kde')
    plt.title('Total reviews')
    st.pyplot(plt)
    
elif box == "Đề xuất sản phẩm khi khách hàng chọn một sản phẩm bất kỳ":

    st.image('image/tiki_banner_1.jpg')
    option_all_users = st.selectbox('Vui lòng chọn sản phẩm bạn cần tìm kiếm:',products['name'].sort_values().unique().tolist())
    st.markdown("### Bạn đã chọn sản phẩm:")
    st.write(option_all_users)
    products_chosen = products.loc[products['name'] == option_all_users]

    img = products_chosen['image'].tolist()
    brand_choose = products_chosen['brand'].tolist()
    price_choose = products_chosen['price'].tolist()
    rating_choose = products_chosen['rating'].tolist()
    col1, col2 = st.columns(2)

    with col1:
        st.image(img[0])
    with col2:
        st.write("Tên sản phẩm:    ",option_all_users)

        st.write("Thương hiệu:    ",brand_choose[0])

        st.write("Giá:    ",f"{price_choose[0]:,}","VND")

        st.write("Đánh giá:   ",str(rating_choose[0]),"/ 5.0 :star:")

    st.markdown("### Các sản phẩm tương tự")

    id_product_chosen = products_chosen['item_id'].tolist()
    product_id = id_product_chosen[0]
    product = products[products.item_id == product_id].head(1)

    name_discription_pre = product['name_description_pre'].to_string(index=False)
    results = recommender(name_discription_pre,dictionary,tfidf,index)
    results = results[results.item_id != product_id]

    col0,col1,col2,col3,col4 = st.columns(5)
    with col0:
        product_recom0 = products.loc[products['item_id'] == results['item_id'].iloc[0]]
        img_recom = product_recom0['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_recom = product_recom0['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_recom[0])
        brand_recom = product_recom0['brand'].tolist()
        st.write("Thương hiệu:",brand_recom[0])
        price_recom = product_recom0['price'].tolist()
        st.write("Giá:",f"{price_recom[0]:,}","VND")
        rating_recom = product_recom0['rating'].tolist()
        st.write("Đánh giá:",str(rating_recom[0]),"/ 5.0 :star:")
        score_recom0 = results['score'].iloc[0]
        st.write("Điểm similarity:",f"{score_recom0:.3f}",":thumbsup:")
    with col1:
        product_recom1 = products.loc[products['item_id'] == results['item_id'].iloc[1]]
        img_recom = product_recom1['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_recom = product_recom1['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_recom[0])
        brand_recom = product_recom1['brand'].tolist()
        st.write("Thương hiệu:",brand_recom[0])
        price_recom = product_recom1['price'].tolist()
        st.write("Giá:",f"{price_recom[0]:,}","VND")
        rating_recom = product_recom1['rating'].tolist()
        st.write("Đánh giá:",str(rating_recom[0]),"/ 5.0 :star:")
        score_recom1 = results['score'].iloc[1]
        st.write("Điểm similarity:",f"{score_recom1:.3f}",":thumbsup:")
    with col2:
        product_recom2 = products.loc[products['item_id'] == results['item_id'].iloc[2]]
        img_recom = product_recom2['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_recom = product_recom2['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_recom[0])
        brand_recom = product_recom2['brand'].tolist()
        st.write("Thương hiệu:",brand_recom[0])
        price_recom = product_recom2['price'].tolist()
        st.write("Giá:",f"{price_recom[0]:,}","VND")
        rating_recom = product_recom2['rating'].tolist()
        st.write("Đánh giá:",str(rating_recom[0]),"/ 5.0 :star:")
        score_recom2 = results['score'].iloc[2]
        st.write("Điểm similarity:",f"{score_recom2:.3f}",":thumbsup:")
    with col3:
        product_recom3 = products.loc[products['item_id'] == results['item_id'].iloc[3]]
        img_recom = product_recom3['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_recom = product_recom3['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_recom[0])
        brand_recom = product_recom3['brand'].tolist()
        st.write("Thương hiệu:",brand_recom[0])
        price_recom = product_recom3['price'].tolist()
        st.write("Giá:",f"{price_recom[0]:,}","VND")
        rating_recom = product_recom3['rating'].tolist()
        st.write("Đánh giá:",str(rating_recom[0]),"/ 5.0 :star:")
        score_recom3 = results['score'].iloc[3]
        st.write("Điểm similarity:",f"{score_recom3:.3f}",":thumbsup:")
    with col4:
        product_recom4 = products.loc[products['item_id'] == results['item_id'].iloc[4]]
        img_recom = product_recom4['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_recom = product_recom4['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_recom[0])
        brand_recom = product_recom4['brand'].tolist()
        st.write("Thương hiệu:",brand_recom[0])
        price_recom = product_recom4['price'].tolist()
        st.write("Giá:",f"{price_recom[0]:,}","VND")
        rating_recom = product_recom4['rating'].tolist()
        st.write("Đánh giá:",str(rating_recom[0]),"/ 5.0 :star:")
        score_recom4 = results['score'].iloc[4]
        st.write("Điểm similarity:",f"{score_recom4:.3f}",":thumbsup:")

elif box == "Đề xuất sản phẩm bằng ID khách hàng":
    # Recommendation for customer_id = list or input by customer
    customer_id = st.text_input("Vui lòng nhập ID:", value= 5682927, max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None)
    reviews_cus = reviews.loc[reviews['customer_id'] == int(customer_id)]
    cus_product = products.loc[products['item_id'].isin(reviews_cus['product_id'])]
    
    st.write("Chào mừng khách hàng :",reviews_cus['name'].unique().tolist()[0])
    st.write("Sản phẩm đã từng mua:")
    new_data_cus = reviews_cus.merge(cus_product, how='inner', left_on='product_id', right_on='item_id')
    final_data_cus = {"Mã khách hàng": new_data_cus.customer_id, "Tên khách hàng" : new_data_cus.name_x,"Mã sản phẩm" : new_data_cus.product_id,"Tên sản phẩm" : new_data_cus.name_y,"Đánh giá" : new_data_cus.rating_x}

    st.table(final_data_cus)

    find_user_rec = recommender_user.filter(recommender_user['customer_id'] == customer_id)

    result = ''
    for user in find_user_rec.collect():
        lst = []
        for row in user['recommendations']:
            print(row)
            lst.append((row['product_id'],row['rating']))
        dic_user_rec = {'customer_id' : user.customer_id, 'recommendations' : lst}
        result = dic_user_rec

    user_result = pd.DataFrame(result)
    user_result[['product_id','rating']] = pd.DataFrame(user_result['recommendations'].tolist(), index= user_result.index)

    st.markdown("#### Các sản phẩm được đề xuất:")

    user_col0,user_col1,user_col2,user_col3,user_col4 = st.columns(5)
    with user_col0:
        product_user0 = products.loc[products['item_id'] == user_result['product_id'].iloc[0]]
        img_recom = product_user0['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_user = product_user0['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_user[0])
        brand_user = product_user0['brand'].tolist()
        st.write("Thương hiệu:",brand_user[0])
        price_user = product_user0['price'].tolist()
        st.write("Giá:",f"{price_user[0]:,}","VND")
        rating_user = product_user0['rating'].tolist()
        st.write("Đánh giá:",str(rating_user[0]),"/ 5.0 :star:")
        score_user0 = user_result['rating'].iloc[0]
        st.write("Điểm similarity:",f"{score_user0:.3f}",":thumbsup:")

    with user_col1:
        product_user1 = products.loc[products['item_id'] == user_result['product_id'].iloc[1]]
        img_recom = product_user1['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_user = product_user1['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_user[0])
        brand_user = product_user1['brand'].tolist()
        st.write("Thương hiệu:",brand_user[0])
        price_user = product_user1['price'].tolist()
        st.write("Giá:",f"{price_user[0]:,}","VND")
        rating_user = product_user1['rating'].tolist()
        st.write("Đánh giá:",str(rating_user[0]),"/ 5.0 :star:")
        score_user1 = user_result['rating'].iloc[1]
        st.write("Điểm similarity:",f"{score_user1:.3f}",":thumbsup:")

    with user_col2:
        product_user2 = products.loc[products['item_id'] == user_result['product_id'].iloc[2]]
        img_recom = product_user2['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_user = product_user2['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_user[0])
        brand_user = product_user2['brand'].tolist()
        st.write("Thương hiệu:",brand_user[0])
        price_user = product_user2['price'].tolist()
        st.write("Giá:",f"{price_user[0]:,}","VND")
        rating_user = product_user2['rating'].tolist()
        st.write("Đánh giá:",str(rating_user[0]),"/ 5.0 :star:")
        score_user2 = user_result['rating'].iloc[2]
        st.write("Điểm similarity:",f"{score_user2:.3f}",":thumbsup:")

    with user_col3:
        product_user3 = products.loc[products['item_id'] == user_result['product_id'].iloc[3]]
        img_recom = product_user3['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_user = product_user3['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_user[0])
        brand_user = product_user3['brand'].tolist()
        st.write("Thương hiệu:",brand_user[0])
        price_user = product_user3['price'].tolist()
        st.write("Giá:",f"{price_user[0]:,}","VND")
        rating_user = product_user3['rating'].tolist()
        st.write("Đánh giá:",str(rating_user[0]),"/ 5.0 :star:")
        score_user3 = user_result['rating'].iloc[3]
        st.write("Điểm similarity:",f"{score_user3:.3f}",":thumbsup:")

    with user_col4:
        product_user4 = products.loc[products['item_id'] == user_result['product_id'].iloc[4]]
        img_recom = product_user4['image'].tolist()
        st.image(img_recom[0],use_column_width=True)
        name_user = product_user4['name'].unique().tolist()
        st.write("Tên sản phẩm:",name_user[0])
        brand_user = product_user4['brand'].tolist()
        st.write("Thương hiệu:",brand_user[0])
        price_user = product_user4['price'].tolist()
        st.write("Giá:",f"{price_user[0]:,}","VND")
        rating_user = product_user4['rating'].tolist()
        st.write("Đánh giá:",str(rating_user[0]),"/ 5.0 :star:")
        score_user4 = user_result['rating'].iloc[4]
        st.write("Điểm similarity:",f"{score_user4:.3f}",":thumbsup:")