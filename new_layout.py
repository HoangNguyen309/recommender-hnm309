import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from recommender import recommender

st.set_page_config(page_title='Hệ thống đề xuất sản phẩm')

st.image("image/csc_banner.png")
st.markdown("<h1 style='text-align: center;background-color:powderblue;'>Đồ án tốt nghiệp Data Science</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Chủ đề: Recommendation System (Tiki.vn)</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Nhóm<br>Trần Trọng Huy - Nguyễn Minh Hoàng</h3>", unsafe_allow_html=True)

products = pd.read_csv("final_product.csv",skipinitialspace=True)
products = products.drop(columns=["Unnamed: 0"]).set_index("index")

reviews = pd.read_csv("final_review.csv",skipinitialspace=True)
reviews = reviews.drop(columns=["Unnamed: 0"]).set_index("id")

dictionary = pickle.load(open('Dictionary.sav', 'rb'))
tfidf = pickle.load(open('TfidfModel.sav', 'rb'))
index = pickle.load(open('Index.sav', 'rb'))

box = st.selectbox("Xin mời lựa chọn:",("Mục tiêu của hệ thống đề xuất sản phẩm","Xây dựng hệ thống","Đề xuất sản phẩm khi khách hàng chọn một sản phẩm bất kỳ","Đề xuất sản phẩm bằng ID khách hàng"))
if box == "Mục tiêu của hệ thống đề xuất sản phẩm":
    st.image('image/RecommendationEngine-1200x675.png')
    st.markdown("<p style='text-align: center;'>Có rất nhiều ứng dụng mà các trang web thu thập dữ liệu từ người dùng của họ và sử dụng dữ liệu đó để dự đoán lượt thích và không thích của người dùng.<br>Hỗ trợ ra quyết định, cung cấp giải pháp mang tính cá nhân hóa mà không phải trải qua quá trình tìm kiếm phức tạp. Điều này cho phép họ giới thiệu nội dung mà họ thích. Thu thập hành vi và dữ liệu người dùng trước và đưa ra các gợi ý các sản phẩm tốt nhất trong số các sản phẩm phù hợp cho người dùng hiện hành.</p>", unsafe_allow_html=True)
    st.image("image/recom_sys.png")
    st.markdown("<p style='text-align: center;'>Có 2 loại hệ thống đề xuất đó là: </p>",unsafe_allow_html=True)
    st.markdown("<span style='text-align: center;'>* Collaborative Filtering: Đề xuất các mục dựa trên sự đo lường mức độ giống nhau giữa người dùng hoặc các sản phẩm. Giả định cơ bản đằng sau thuật toán là những người dùng có cùng sở thích sẽ có chung sở thích.</span>",unsafe_allow_html=True)
    st.markdown("<span style='text-align: center;'>* Content-Based Recommendation: Công nghệ máy học có giám sát được sử dụng để nhận biết và phân biệt giữa các mục hoặc sản phẩm thú vị và không thú vị đối với người dùng.</p>",unsafe_allow_html=True)
    st.image("image/recom_sys_2.png")
    st.markdown("<p style='text-align: center;'>* Content-Based Recommendation System: Hệ thống dựa trên nội dung đề xuất các mặt hàng cho khách hàng tương tự như các mặt hàng đã được khách hàng xếp hạng cao trước đó. Nó sử dụng các tính năng và thuộc tính của mặt hàng. Từ các thuộc tính này, nó có thể tính toán mức độ giống nhau giữa các mục.</p>",unsafe_allow_html=True)

elif box == "Xây dựng hệ thống":
    st.image("image/toptal-blog-image.png")

    st.write("""
    #### Dữ liệu về phần sản phẩm:
    """)
    st.dataframe(products.head())
    st.write("""
    #### Dữ liệu về phần nhận xét:
    """)
    st.dataframe(reviews.head())

    st.write("""
    #### Tìm hiểu thêm về bộ dữ liệu:
    """)
    st.dataframe(products[['price','list_price','rating']].describe())
    st.markdown("""
    * 'price' và 'list price' có giá trị trong khoảng 7000 đến ~62.7 triệu
    * 'rating' trong khoảng từ 0-5
    """)

    brands = products.groupby('brand')['item_id'].count().sort_values(ascending=False)
    plt.subplots_adjust(top=1,bottom=0)
    brands[1:11].plot(kind='bar')
    plt.ylabel('số lượng')
    plt.title('Số lượng sản phẩm tính theo nhãn hàng')
    st.pyplot(plt)

    st.markdown("""
    => Samsung là nhãn hàng có số lượng sản phẩm nhiều nhất.
    """)

    plt.subplots_adjust(top=1,bottom=0)
    price_by_brand = products.groupby(by='brand').mean()['price']
    price_by_brand.sort_values(ascending=False)[:10].plot(kind='bar')
    plt.ylabel('giá')
    plt.title('Giá trị trung bình tính theo nhãn hàng')
    st.pyplot(plt)
    
    st.markdown("""
    => Thương hiệu Hitachi có giá trung bình cao nhất.
    """)

    # Top 20 products have most positive review
    plt.figure(figsize=(10,4))
    top_products = reviews.groupby('product_id').count()['customer_id'].sort_values(ascending=False)[:20]
    top_products.index = products[products.item_id.isin(top_products.index)]['name'].str[:25]
    top_products.plot(kind='bar')
    plt.title('Top 20 sản phẩm nhận được nhiều lượt đánh giá tích cực')
    st.pyplot(plt)

    st.markdown("""
    => Chuột không giây Logitech nhận được nhiều lượt đánh giá tích cực nhất.
    """)

    # Top 20 customer make review
    top_rating_customers = reviews.groupby('customer_id').count()['product_id'].sort_values(ascending=False)[:20]
    plt.figure(figsize=(12,6))
    plt.bar(x=[str(x) for x in top_rating_customers.index],height=top_rating_customers.values)
    plt.xticks(rotation=60)
    plt.title('Top 20 khách hàng có lượt đánh giá về sản phẩm nhiều nhất')
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

    csv_recommender_user = pd.read_csv("user_recs.csv")
    csv_recommender_user = csv_recommender_user.drop(columns="Unnamed: 0")

    csv_recommender_user_final = csv_recommender_user.loc[csv_recommender_user['customer_id']==int(customer_id)]

    st.markdown("#### Các sản phẩm được đề xuất:")

    user_col0,user_col1,user_col2,user_col3,user_col4 = st.columns(5)
    with user_col0:
        product_user0 = products.loc[products['item_id'] == csv_recommender_user_final['product_id'].iloc[0]]
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
        score_user0 = csv_recommender_user_final['rating'].iloc[0]
        st.write("Điểm similarity:",f"{score_user0:.3f}",":thumbsup:")

    with user_col1:
        product_user1 = products.loc[products['item_id'] == csv_recommender_user_final['product_id'].iloc[1]]
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
        score_user1 = csv_recommender_user_final['rating'].iloc[1]
        st.write("Điểm similarity:",f"{score_user1:.3f}",":thumbsup:")

    with user_col2:
        product_user2 = products.loc[products['item_id'] == csv_recommender_user_final['product_id'].iloc[2]]
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
        score_user2 = csv_recommender_user_final['rating'].iloc[2]
        st.write("Điểm similarity:",f"{score_user2:.3f}",":thumbsup:")

    with user_col3:
        product_user3 = products.loc[products['item_id'] == csv_recommender_user_final['product_id'].iloc[3]]
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
        score_user3 = csv_recommender_user_final['rating'].iloc[3]
        st.write("Điểm similarity:",f"{score_user3:.3f}",":thumbsup:")

    with user_col4:
        product_user4 = products.loc[products['item_id'] == csv_recommender_user_final['product_id'].iloc[4]]
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
        score_user4 = csv_recommender_user_final['rating'].iloc[4]
        st.write("Điểm similarity:",f"{score_user4:.3f}",":thumbsup:")