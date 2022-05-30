import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as scpy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

indexRetrofit = ['Mevcut Durum','KLP']
columnsRetrofit = ['Mevcut Maliyet', 'Güçlendirme Maliyeti','Beklenen Maliyet', 'Toplam Maliyet']
buildingsComponents = ['Structural', 'NonStructural','MajorAppliances', 'HumanLoss']

comparisonOfRefrofitMethods = pd.DataFrame(np.zeros((1,len(columnsRetrofit))), index = [indexRetrofit[0]],columns=columnsRetrofit)
# comparisonOfRefrofitMethods=comparisonOfRefrofitMethods.drop(comparisonOfRefrofitMethods.index[indexRetrofit[0]])
comparisonOfBuildingComponents = pd.DataFrame(np.zeros((1,len(buildingsComponents))), index = [indexRetrofit[0]],columns=buildingsComponents)
 
    
# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 3
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Projects", "Contact"],  # required
                icons=["house", "building", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Contact"],  # required
            icons=["house", "building", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Contact"],  # required
            icons=["house", "building", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

def plotFragilityCurveofEachState(maksXaxes=80,title='Kırılganlık Eğrileri'): 
    st.subheader(title)
    figFragilityCurve = go.Figure()
    # Create and style traces
    figFragilityCurve.add_trace(go.Scatter(x=edp, y=cdf[:,0], name='SH',
                             line=dict(color='green', width=2)))
    figFragilityCurve.add_trace(go.Scatter(x=edp, y=cdf[:,1], name='KH',
                         line=dict(color='royalblue', width=2)))

    figFragilityCurve.add_trace(go.Scatter(x=edp, y=cdf[:,2], name='GÖ',
                         line=dict(color='firebrick', width=2)))
    
    # figFragilityCurve.add_trace(go.Scatter(x=edp, y=cdf[:,3], name='CP',
    #                      line=dict(color='continent', width=4)))
    
    # Edit the layout
    figFragilityCurve.update_layout(
    legend = dict(
        font = dict(size = 15, color = "black")),
        margin=dict(l=15, r=15, t=22, b=20),
        xaxis=dict(
        title='PGV (cm/s)',
        titlefont_size=16,
        tickfont_size=14),
        yaxis=dict(
        title='Aşılma Olasılığı',
        titlefont_size=16,
        tickfont_size=14),height=400,width=700,uniformtext_minsize=12, uniformtext_mode='hide')
        
    figFragilityCurve.update_xaxes(range=[2,maksXaxes],  # sets the range of xaxis,
                                       constrain="domain"  # meanwhile compresses the xaxis by decreasing its "domain"
                                       )
    figFragilityCurve.update_yaxes(range=[0,1])

    st.markdown('Hasar Seviyeleri için **_Kırılganlık Eğrileri_**')
    st.write(figFragilityCurve)
  
  
  
def makeTableAndChartofExpectedCost(title='Beklenen Kayıplar'): 
    st.subheader(title)
    fig4 = make_subplots(
        rows=1, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}, {"type": "bar"}]],
    
    )
    
    
    fig4.add_trace(go.Table(
            domain=dict(x=[0, 0.5],
                        y=[0, 1.0]),
            columnwidth = [2,3],
            # columnorder=[0, 1, 2, 3, 4],
            header=dict(height = 30,
                        values=['Toplam Yıl', 'Kayıp ($)'], 
                        line = dict(color='rgb(50, 50, 50)'),
                        align = ['center'] * 5,
                        font = dict(color = 'darkslategray', size = 16)),
    
            cells=dict(values=[table['Year'], round(table['Cost'],1)],
                        line = dict(color='#506784'),
                        align = ['center'] * 5,
                        font = dict(color=['rgb(40, 40, 40)'] * 5, size=16),
                        height = 27)),
                    1,1)
    
    fig4.add_trace(go.Bar(   
        x=table['Year'],
        y=table['Cost'],
        name='Expected Cost',
        marker_color='indianred',
        marker=dict(coloraxis="coloraxis"),
        width=[1.8]*len(table),     
        textposition='auto'),
        1,2)
    
    fig4.update_xaxes(tickvals=table['Year'].values, title_text='Yıl')
    fig4.update_yaxes(title_text='Kayıp ($)')
    fig4.update_layout(height=300,width=1000, showlegend=True)
    fig4.update_layout(margin=dict(l=5,r=5,b=5,t=5))

    st.markdown('Binalarda belirli bir zaman dilimi için **_beklenen kayıpların_** tahmini ')
    st.write(fig4)    

        
def obtainTableofExpectedCost():
    fig3 = go.Figure(data=[go.Table(
        header=dict(height=30,
                    values=['Total Year', 'Cost'], font = dict(color = 'darkslategray', size = 16)),
                                    
        cells=dict(height=27,values=[table['Year'], round(table['Cost'],1)],
                   align=['center','center'], 
                   fill_color='lightgrey', 
                   font = dict(color = 'darkslategray', size = 16)  ))
                           ])
    fig3.update_layout(title_xanchor="left", width=450, height=550, 
                       title_font=dict(color ='darkslategray', size = 20),    
                       title={'text': "Beklenen Maliyetlerin Yıllara Göre Değeri",
                                'y':0.91,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'} )
                        
    st.write(fig3)
    
# =============================================================================
# HOME PAGE
# =============================================================================

if selected == "Home":
    # About
    expander_bar = st.expander("About")
    expander_bar.markdown("""
    * **Python kütüphaneleri:** pandas, streamlit, numpy, matplotlib, plotly, os, scipy
    * **Data source:** [CoinMarketCap](http://coinmarketcap.com).
    * **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
    """)
    st.write("""
    **NOTLAR** 
    * Expected annual loss - bar chart çizimi ekle, hangi hasar seviyesinin en çok katkı yaptığını göster
    * upload hazard değerleri interpolasyon yapılacak şekilde ayarlanmalı, sadece default hız için düzgün çalışacak şu an
    """)  
    st.write("""
    --------------------------------------------------------------------------------------------------------------------------
    ÖZLEM ÖZDERYA seni seviyorum
    """)  

# =============================================================================
# EXPECTED COST OF BUILDINGS
# =============================================================================
if selected == "Projects":
    sdasdas = []

    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.sidebar.header('Tehlike Eğrisi Değerleri')
    uploaded_file = st.sidebar.file_uploader("Saha için değerleri yükleyin", type=["csv"])
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)
    
    #  **** HAZARD ****
    if uploaded_file is not None:
        # hazardData = pd.read_csv(uploaded_file)  düzenlenecek
        print(s)
    else:
        hazardData50 = pd.read_csv('data.csv')
        hazardData50 = pd.DataFrame(hazardData50)
        hazardData50.columns=['Velocity','Hazard']

        data1 = hazardData50['Velocity']
        data2 = hazardData50['Hazard']
        
        # dataları log uzayına çevirme
        pgv = np.log10(data1)
        probability = np.log10(data2)
        f = interp1d(pgv,probability)
        data_pgv = np.arange(2,150,1)
        pgv = np.log10(data_pgv)
        P = f(pgv)    
        edp = (10**pgv)
        hazard = (10**P/50)
        annualHazard = pd.DataFrame({'EDP': edp, 'Probability': hazard}, columns=['EDP', 'Probability'])

    
    Navigation_Retrofit=["Mevcut Durum","KLP Kompozit ile Sarma","Sonuçları Karşılaştırma"]
    rad =st.sidebar.radio("Navigation",Navigation_Retrofit)

    #*******************Mevcut Durum***********************************************
    if rad == Navigation_Retrofit[0]:
        st.header("Binanın Mevcut Durumu")
    
        st.sidebar.header('Kırılganlık Parametreleri')
        
        with st.sidebar.form(key='teta değerleri'):
            teta = st.text_input('Median Value',placeholder='Örnek: 13.0, 15.5, 21.0')
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""
        * **bilgi ver:** """)
        
        with st.sidebar.form(key='beta değerleri'):
            beta = st.text_input('Standart Deviation',placeholder='Örnek: 0.25, 0.3, 0.3')
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""* **bilgi ver:** """)
        
              
        # **** CAPACITY ****
        if teta != "" and beta!="":
            teta = np.fromstring(teta, dtype=float, sep=',')
            beta = np.fromstring(beta, dtype=float, sep=',')
            if len(teta)==4:
                showTeta = pd.DataFrame(teta,index=['OP','IO','LS','CP'], columns=['Median Values'])
            else:
                showTeta = pd.DataFrame(teta,index=['IO','LS','CP'], columns=['Median Values'])
               
            #st.write(showTeta)

        else:
                           # slight, moderate, major, collapse
            teta = np.array([13.0, 15.5, 21])
            beta = np.array([0.35, 0.33, 0.26])
            #st.write(showTeta)


        st.sidebar.header('Maliyet Parametreleri')
        component = st.sidebar.multiselect('Hesaplara dahil edilecek bileşenleri seçin', ['Structural','Nonstructural','Major Appliances','Human Life'],default=('Structural','Nonstructural'))
        component = np.array(component)
        H = component[np.where(component=='Human Life')]
        S = component[np.where(component=='Structural')]
        N = component[np.where(component=='Nonstructural')]
        A = component[np.where(component=='Major Appliances')]

        metrekareOran=894.6/3705
        defaultStructuralCostInitial = round(8971716*metrekareOran/14.57)   # dolar kuru 14.57 , maliyet 8971716
        defaultNonstructuralCostInitial = round(11768794*metrekareOran/14.57)
        defaultHumanLifeCostInitial = 0
        defaultMajorCostInitial = 0
        defaultDiscountRate = 0
        with st.sidebar.form(key='maliyet değerleri'):
            discountRate = st.number_input('Discoun Rate (%)',value=defaultDiscountRate,step =1)
        
            if S.size!=0:
                structural_initial_cost = st.number_input('Structural initial cost',value=defaultStructuralCostInitial,step =10)
            else:
                structural_initial_cost=0
            if N.size!=0:
                nonstructural_initial_cost = st.number_input('Nontructural initial cost',value=defaultNonstructuralCostInitial,step =10)
            else:
                nonstructural_initial_cost=0
            if H.size!=0:
                humanlife_initial_cost = st.number_input('Human Life initial cost',value=defaultHumanLifeCostInitial,step =10)
                numberofLossLife = st.number_input('Number of Life Loss',value=0,step =1)
                humanlife_initial_cost = numberofLossLife*humanlife_initial_cost  # Human Life impact of the structure
            else:
                humanlife_initial_cost=0
            if A.size!=0:
                majorAppliance_initial_cost = st.number_input('Major Appliance initial cost',value=defaultMajorCostInitial,step =10)
            else:
                majorAppliance_initial_cost=0
            
            totalInitialCostOfAllComponent = structural_initial_cost+nonstructural_initial_cost+majorAppliance_initial_cost
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""* **bilgi ver:** """)
        
        
        # ****IMPACT ****
        cdf = np.zeros((len(edp),len(teta)))
        for i in range(len(teta)):
            val = scpy.norm.cdf(np.log(edp),np.log(teta[i]), beta[i])
            cdf[:,i] = val
        
        # the probability of only event Ei occuring for a given PGA value a    
        oneEventPDF = np.zeros((len(edp),len(teta)))
        for i in range(len(teta)-1):
            oneEventPDF[:,i] = cdf[:,i]- cdf[:,i+1]   
        oneEventPDF[:,len(teta)-1] = cdf[:,len(teta)-1]
        
        # the probability that no earthquake has happenned in the previous T-1 years
        maxYear = 100
        Tyear = np.linspace(1,maxYear,maxYear)
        
        hazardForMin = annualHazard['Probability'][0]   # R(amin) = 0.019798   for  exp(-R(amin)*(T-1))
        listNoEQ = np.zeros((maxYear,1))
        
        for t in range(1,maxYear+1):
            listNoEQ[t-1] = np.exp(-hazardForMin*(t-1))
        
        # the probability of exceeding the PGA value a given that no earthquake has 
        # occurred in the previous (T-1) years.  R(a,T)
        
        for t in range(1,maxYear+1):
            RaT = annualHazard['Probability'].to_numpy()*(listNoEQ)
            RaT = np.transpose(RaT)
        
        # kg CO2 for damage states Ei
        discountRate = discountRate/100   # % 0
        
        h = [0.0,0.0,0.0,1]  # % 0, % 0, % 0, % 100
        # structural_initial_cost = 1733633.8  # structural impact of the structure
        s = [0.0,0.15,0.6,1]  # % 0, % 15, % 60, % 100
        # nonstructural_initial_cost = 656814.8  # Non-structural impact of the structure
        n = [0.15,0.35,0.85,1]  # % 15, % 35, % 85, % 100
        # majorAppliance_initial_cost = 56827.44  # Major appliances impact of the structure
        a = [0.0,0.10,0.5,1]  # % 0, % 10, % 50, % 100
        
        Tco2 = np.zeros((maxYear,len(teta)))
        Tco2Struc = np.zeros((maxYear,len(teta)))
        Tco2NonStruc = np.zeros((maxYear,len(teta)))
        Tco2Human = np.zeros((maxYear,len(teta)))
        Tco2Major = np.zeros((maxYear,len(teta)))
        for i in range(len(teta)):
            for t in range(1,maxYear+1):
                if len(teta)==3 and len(beta)==3:
                    Tco2Struc[t-1,i] =  structural_initial_cost * s[i+1] /(1+discountRate)**(t-1)
                    Tco2NonStruc[t-1,i] =  nonstructural_initial_cost * n[i+1] /(1+discountRate)**(t-1)
                    Tco2Major[t-1,i] = majorAppliance_initial_cost * a[i+1] /(1+discountRate)**(t-1)
                    Tco2Human[t-1,i] =  humanlife_initial_cost * h[i+1] /(1+discountRate)**(t-1)
                    #Tco2[t-1,i] = humanlife_initial_cost * h[i+1] /(1+discountRate)**(t-1) + structural_initial_cost * s[i+1] /(1+discountRate)**(t-1) + nonstructural_initial_cost * n[i+1] /(1+discountRate)**(t-1) + majorAppliance_initial_cost * a[i+1] /(1+discountRate)**(t-1)
                else:
                    #Tco2[t-1,i] = humanlife_initial_cost * h[i] /(1+discountRate)**(t-1) + structural_initial_cost * s[i] /(1+discountRate)**(t-1) + nonstructural_initial_cost * n[i] /(1+discountRate)**(t-1) + majorAppliance_initial_cost * a[i] /(1+discountRate)**(t-1)
                    Tco2Struc[t-1,i] =  structural_initial_cost * s[i] /(1+discountRate)**(t-1)
                    Tco2NonStruc[t-1,i] =  nonstructural_initial_cost * n[i] /(1+discountRate)**(t-1)
                    Tco2Major[t-1,i] = majorAppliance_initial_cost * a[i] /(1+discountRate)**(t-1)
                    Tco2Human[t-1,i] =  humanlife_initial_cost * h[i] /(1+discountRate)**(t-1)
        
        Tco2 = Tco2Major+Tco2Human+Tco2NonStruc+Tco2Struc

        # expected cost multiplier        
        damageCostT = np.zeros((maxYear,2))
        expectedMultiplier = np.zeros((len(edp)-1,len(teta)))
        damageCostTS = np.zeros((maxYear,2))
        damageCostTN = np.zeros((maxYear,2))
        damageCostTA = np.zeros((maxYear,2))
        damageCostTH = np.zeros((maxYear,2))
        
        for t in range(1,maxYear+1):
            for i in range(len(teta)):
                for k in range(len(edp)-1):
                    expectedMultiplier[k,i]= 0.5*(oneEventPDF[k,i]+oneEventPDF[k+1,i])*(RaT[k,t-1]-RaT[k+1,t-1])
                    columnSum = expectedMultiplier.sum(axis=0)
                    cost = np.dot(Tco2[t-1,:],columnSum)
                    damageCostT[t-1,0] = t
                    damageCostT[t-1,1] = cost
                    
                    costS = np.dot(Tco2Struc[t-1,:],columnSum)
                    damageCostTS[t-1,0] = t
                    damageCostTS[t-1,1] = costS
                    costN = np.dot(Tco2NonStruc[t-1,:],columnSum)
                    damageCostTN[t-1,0] = t
                    damageCostTN[t-1,1] = costN
                    costA = np.dot(Tco2Major[t-1,:],columnSum)
                    damageCostTA[t-1,0] = t
                    damageCostTA[t-1,1] = costA
                    costH = np.dot(Tco2Human[t-1,:],columnSum)           
                    damageCostTH[t-1,0] = t
                    damageCostTH[t-1,1] = costH
                    
        # T* yıl için COST 
        for i in range(maxYear-1):
            delta = damageCostT[i,1] + damageCostT[i+1,1]
            damageCostT[i+1,1] = delta
            deltaS = damageCostTS[i,1] + damageCostTS[i+1,1]
            damageCostTS[i+1,1] = deltaS         
            deltaN = damageCostTN[i,1] + damageCostTN[i+1,1]
            damageCostTN[i+1,1] = deltaN
            deltaA = damageCostTA[i,1] + damageCostTA[i+1,1]
            damageCostTA[i+1,1] = deltaA
            deltaH = damageCostTH[i,1] + damageCostTH[i+1,1]
            damageCostTH[i+1,1] = deltaH
        
        damageCostTyear = pd.DataFrame(damageCostT,columns= ['Year','Cost'])
        damageCostTyearS = pd.DataFrame(damageCostTS,columns= ['Year','Cost'])
        damageCostTyearN = pd.DataFrame(damageCostTN,columns= ['Year','Cost'])
        damageCostTyearA = pd.DataFrame(damageCostTA,columns= ['Year','Cost'])
        damageCostTyearH = pd.DataFrame(damageCostTH,columns= ['Year','Cost'])

        if 'buildingsComponents' not in st.session_state:
            st.session_state.buildingsComponents = comparisonOfBuildingComponents
        st.session_state.buildingsComponents.loc[[indexRetrofit[0]]] = [damageCostTyearS.iloc[49,1],damageCostTyearN.iloc[49,1],damageCostTyearA.iloc[49,1],damageCostTyearH.iloc[49,1]]
                
        
        st.sidebar.header('Bina kullanım ömrü')
        buildingUseLife = st.sidebar.multiselect('Varsayılan bina kullanım ömrünü seçin (yıl)', ['1' ,'5','10 ','25','50', '75'],default=('1','5','10 ','25','50','75'))
        buildingUseLife = np.array(buildingUseLife).astype(int)
        np.set_string_function(lambda x: repr(buildingUseLife), repr=False)
        table = damageCostTyear.iloc[buildingUseLife-1,0:2]
        
        # GÜÇÇLENDİRME MALİYETİ TANIMLAMA
        st.sidebar.header('Güçlendirme Maliyeti')
        
        with st.sidebar.form(key='güçlendirme maliyet değerleri'):
            retrofitMD = st.number_input('Mevcut Durum Güçlendirme',value=0,step =1)
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""
        * **bilgi ver:** """)
        
        
       # Write Streamlit Page 
        plotFragilityCurveofEachState()   # kırılganlık eğrilerinin çizilmesi

        

        if len(buildingUseLife)>0:
            makeTableAndChartofExpectedCost()
        
        
                
        # Verilen durum için elde edilen sonuçları kıyaslamak için kaydetme !!!
        with st.form(key='Kıyaslama için Ekle'):
            st.write("Bina kullanım ömrünün **50 yıl** olduğu durum için sonuçları verir")
            submitForComparisonMD = st.form_submit_button(label='Submit')         
            if submitForComparisonMD:
                if 'MD' not in st.session_state:
                    st.session_state.MD = comparisonOfRefrofitMethods
                
                # st.write(st.session_state.MD.damageCostTyear.iloc[49,1])
                comparisonOfRefrofitMethods.loc[[indexRetrofit[0]]] = [totalInitialCostOfAllComponent,retrofitMD,damageCostTyear.iloc[49,1],totalInitialCostOfAllComponent+retrofitMD+damageCostTyear.iloc[49,1]]
                st.session_state.MD = st.session_state.MD.append(comparisonOfRefrofitMethods)
        # submite tıklandığında duplicate olmasın diye aşağıdaki yapıldı            
                st.session_state.MD=st.session_state.MD[~st.session_state.MD.index.duplicated(keep='last')]
                st.dataframe(st.session_state.MD.style.format('{:.2f}'))
             
              
     
    #*******************KLP Kompozit ile Güçlendirme***************************************
    if rad == Navigation_Retrofit[1]:
        st.header("KLP Kompozit ile Güçlendirme")
    
        st.sidebar.header('Kırılganlık Parametreleri')
        
        with st.sidebar.form(key='teta değerleri'):
            teta = st.text_input('Median Value',placeholder='Örnek: 13.0, 40.0, 50.0')
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""
        * **bilgi ver:** """)
        
        with st.sidebar.form(key='beta değerleri'):
            beta = st.text_input('Standart Deviation',placeholder='Örnek: 0.25,0.33,0.33')
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""* **bilgi ver:** """)
        
              
        # **** CAPACITY ****
        if teta != "" and beta!="":
            teta = np.fromstring(teta, dtype=float, sep=',')
            beta = np.fromstring(beta, dtype=float, sep=',')
            if len(teta)==4:
                showTeta = pd.DataFrame(teta,index=['OP','IO','LS','CP'], columns=['Median Values'])
            else:
                showTeta = pd.DataFrame(teta,index=['IO','LS','CP'], columns=['Median Values'])
               
            st.write(showTeta)

        else:
                           # slight, moderate, major, collapse
            teta = np.array([13, 40, 50])
            beta = np.array([0.35, 0.33, 0.26])


        st.sidebar.header('Maliyet Parametreleri')
        component = st.sidebar.multiselect('Hesaplara dahil edilecek bileşenleri seçin', ['Structural','Nonstructural','Major Appliances','Human Life'],default=('Structural','Nonstructural'))
        component = np.array(component)
        H = component[np.where(component=='Human Life')]
        S = component[np.where(component=='Structural')]
        N = component[np.where(component=='Nonstructural')]
        A = component[np.where(component=='Major Appliances')]

        metrekareOran=894.6/3705
        defaultStructuralCostInitial = round(8971716*metrekareOran/14.57)   # dolar kuru 14.57 , maliyet 8971716
        defaultNonstructuralCostInitial = round(11768794*metrekareOran/14.57)
        defaultHumanLifeCostInitial = 0
        defaultMajorCostInitial = 0
        defaultDiscountRate = 0
        with st.sidebar.form(key='maliyet değerleri'):
            discountRate = st.number_input('Discoun Rate (%)',value=defaultDiscountRate,step =1)
        
            if S.size!=0:
                structural_initial_cost = st.number_input('Structural initial cost',value=defaultStructuralCostInitial,step =10)
            else:
                structural_initial_cost=0
            if N.size!=0:
                nonstructural_initial_cost = st.number_input('Nontructural initial cost',value=defaultNonstructuralCostInitial,step =10)
            else:
                nonstructural_initial_cost=0
            if H.size!=0:
                humanlife_initial_cost = st.number_input('Human Life initial cost',value=defaultHumanLifeCostInitial,step =10)
                numberofLossLife = st.number_input('Number of Life Loss',value=0,step =1)
                humanlife_initial_cost = numberofLossLife*humanlife_initial_cost  # Human Life impact of the structure
            else:
                humanlife_initial_cost=0
            if A.size!=0:
                majorAppliance_initial_cost = st.number_input('Major Appliance initial cost',value=defaultMajorCostInitial,step =10)
            else:
                majorAppliance_initial_cost=0
            
            totalInitialCostOfAllComponent = structural_initial_cost+nonstructural_initial_cost+majorAppliance_initial_cost
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""* **bilgi ver:** """)
        
        
        # ****IMPACT ****
        cdf = np.zeros((len(edp),len(teta)))
        for i in range(len(teta)):
            val = scpy.norm.cdf(np.log(edp),np.log(teta[i]), beta[i])
            cdf[:,i] = val
        
        # the probability of only event Ei occuring for a given PGA value a    
        oneEventPDF = np.zeros((len(edp),len(teta)))
        for i in range(len(teta)-1):
            oneEventPDF[:,i] = cdf[:,i]- cdf[:,i+1]   
        oneEventPDF[:,len(teta)-1] = cdf[:,len(teta)-1]
        
        # the probability that no earthquake has happenned in the previous T-1 years
        maxYear = 100
        Tyear = np.linspace(1,maxYear,maxYear)
        
        hazardForMin = annualHazard['Probability'][0]   # R(amin) = 0.019798   for  exp(-R(amin)*(T-1))
        listNoEQ = np.zeros((maxYear,1))
        
        for t in range(1,maxYear+1):
            listNoEQ[t-1] = np.exp(-hazardForMin*(t-1))
        
        # the probability of exceeding the PGA value a given that no earthquake has 
        # occurred in the previous (T-1) years.  R(a,T)
        
        for t in range(1,maxYear+1):
            RaT = annualHazard['Probability'].to_numpy()*(listNoEQ)
            RaT = np.transpose(RaT)
        
        # kg CO2 for damage states Ei
        discountRate = discountRate/100   # % 0
        
        h = [0.0,0.0,0.0,1]  # % 0, % 0, % 0, % 100
        # structural_initial_cost = 1733633.8  # structural impact of the structure
        s = [0.0,0.15,0.6,1]  # % 0, % 15, % 60, % 100
        # nonstructural_initial_cost = 656814.8  # Non-structural impact of the structure
        n = [0.15,0.35,0.85,1]  # % 15, % 35, % 85, % 100
        # majorAppliance_initial_cost = 56827.44  # Major appliances impact of the structure
        a = [0.0,0.10,0.5,1]  # % 0, % 10, % 50, % 100
        
        Tco2 = np.zeros((maxYear,len(teta)))
        Tco2Struc = np.zeros((maxYear,len(teta)))
        Tco2NonStruc = np.zeros((maxYear,len(teta)))
        Tco2Human = np.zeros((maxYear,len(teta)))
        Tco2Major = np.zeros((maxYear,len(teta)))
        for i in range(len(teta)):
            for t in range(1,maxYear+1):
                if len(teta)==3 and len(beta)==3:
                    Tco2Struc[t-1,i] =  structural_initial_cost * s[i+1] /(1+discountRate)**(t-1)
                    Tco2NonStruc[t-1,i] =  nonstructural_initial_cost * n[i+1] /(1+discountRate)**(t-1)
                    Tco2Major[t-1,i] = majorAppliance_initial_cost * a[i+1] /(1+discountRate)**(t-1)
                    Tco2Human[t-1,i] =  humanlife_initial_cost * h[i+1] /(1+discountRate)**(t-1)
                    #Tco2[t-1,i] = humanlife_initial_cost * h[i+1] /(1+discountRate)**(t-1) + structural_initial_cost * s[i+1] /(1+discountRate)**(t-1) + nonstructural_initial_cost * n[i+1] /(1+discountRate)**(t-1) + majorAppliance_initial_cost * a[i+1] /(1+discountRate)**(t-1)
                else:
                    #Tco2[t-1,i] = humanlife_initial_cost * h[i] /(1+discountRate)**(t-1) + structural_initial_cost * s[i] /(1+discountRate)**(t-1) + nonstructural_initial_cost * n[i] /(1+discountRate)**(t-1) + majorAppliance_initial_cost * a[i] /(1+discountRate)**(t-1)
                    Tco2Struc[t-1,i] =  structural_initial_cost * s[i] /(1+discountRate)**(t-1)
                    Tco2NonStruc[t-1,i] =  nonstructural_initial_cost * n[i] /(1+discountRate)**(t-1)
                    Tco2Major[t-1,i] = majorAppliance_initial_cost * a[i] /(1+discountRate)**(t-1)
                    Tco2Human[t-1,i] =  humanlife_initial_cost * h[i] /(1+discountRate)**(t-1)
        
        Tco2 = Tco2Major+Tco2Human+Tco2NonStruc+Tco2Struc
        
        
        # expected cost multiplier        
        damageCostT = np.zeros((maxYear,2))
        expectedMultiplier = np.zeros((len(edp)-1,len(teta)))
        damageCostTS = np.zeros((maxYear,2))
        damageCostTN = np.zeros((maxYear,2))
        damageCostTA = np.zeros((maxYear,2))
        damageCostTH = np.zeros((maxYear,2))

        for t in range(1,maxYear+1):
            for i in range(len(teta)):
                for k in range(len(edp)-1):
                    expectedMultiplier[k,i]= 0.5*(oneEventPDF[k,i]+oneEventPDF[k+1,i])*(RaT[k,t-1]-RaT[k+1,t-1])
                    columnSum = expectedMultiplier.sum(axis=0)
                    cost = np.dot(Tco2[t-1,:],columnSum)
                    damageCostT[t-1,0] = t
                    damageCostT[t-1,1] = cost
                    
                    costS = np.dot(Tco2Struc[t-1,:],columnSum)
                    damageCostTS[t-1,0] = t
                    damageCostTS[t-1,1] = costS
                    costN = np.dot(Tco2NonStruc[t-1,:],columnSum)
                    damageCostTN[t-1,0] = t
                    damageCostTN[t-1,1] = costN
                    costA = np.dot(Tco2Major[t-1,:],columnSum)
                    damageCostTA[t-1,0] = t
                    damageCostTA[t-1,1] = costA
                    costH = np.dot(Tco2Human[t-1,:],columnSum)           
                    damageCostTH[t-1,0] = t
                    damageCostTH[t-1,1] = costH
                    
        # T* yıl için COST 
        for i in range(maxYear-1):
            delta = damageCostT[i,1] + damageCostT[i+1,1]
            damageCostT[i+1,1] = delta
            deltaS = damageCostTS[i,1] + damageCostTS[i+1,1]
            damageCostTS[i+1,1] = deltaS         
            deltaN = damageCostTN[i,1] + damageCostTN[i+1,1]
            damageCostTN[i+1,1] = deltaN
            deltaA = damageCostTA[i,1] + damageCostTA[i+1,1]
            damageCostTA[i+1,1] = deltaA
            deltaH = damageCostTH[i,1] + damageCostTH[i+1,1]
            damageCostTH[i+1,1] = deltaH
        
        damageCostTyear = pd.DataFrame(damageCostT,columns= ['Year','Cost'])
        damageCostTyearS = pd.DataFrame(damageCostTS,columns= ['Year','Cost'])
        damageCostTyearN = pd.DataFrame(damageCostTN,columns= ['Year','Cost'])
        damageCostTyearA = pd.DataFrame(damageCostTA,columns= ['Year','Cost'])
        damageCostTyearH = pd.DataFrame(damageCostTH,columns= ['Year','Cost'])
        
        st.session_state.buildingsComponents.loc[[indexRetrofit[1]][0]] = [damageCostTyearS.iloc[49,1],damageCostTyearN.iloc[49,1],damageCostTyearA.iloc[49,1],damageCostTyearH.iloc[49,1]]
    
        
        st.sidebar.header('Bina kullanım ömrü')
        buildingUseLife = st.sidebar.multiselect('Varsayılan bina kullanım ömrünü seçin (yıl)', ['1' ,'5','10 ','25','50', '75'],default=('1','5','10 ','25','50','75'))
        buildingUseLife = np.array(buildingUseLife).astype(int)
        np.set_string_function(lambda x: repr(buildingUseLife), repr=False)
        table = damageCostTyear.iloc[buildingUseLife-1,0:2]
        
        # GÜÇLENDİRME MALİYETİ TANIMLAMA
        st.sidebar.header('Güçlendirme Maliyeti')
        
        with st.sidebar.form(key='güçlendirme maliyet değerleri_KLP'):
            retrofitBM = st.number_input('KLP Kompozit ile Güçlendirme',value=200000,step =1)
            submit_button = st.form_submit_button('Submit')
            expander_bar = st.expander("Açıklama")
            expander_bar.markdown("""
        * **bilgi ver:** """)
        
        
       # SAYFA GİRDİLERİ - GRAFİKLER 
       # Write Streamlit Page 
        plotFragilityCurveofEachState(150)   # kırılganlık eğrilerinin çizilmesi
        
        if len(buildingUseLife)>0:
            makeTableAndChartofExpectedCost()
            
                
        # Verilen durum için elde edilen sonuçları kıyaslamak için kaydetme !!!
        with st.form(key='Kıyaslama için Ekle_KLP'):
            st.write("Bina kullanım ömrünün **50 yıl** olduğu durum için sonuçları verir")
            submitForComparisonMD = st.form_submit_button(label='Submit')         
            if submitForComparisonMD:
                st.session_state.MD.loc[[indexRetrofit[1]][0]] = [totalInitialCostOfAllComponent,retrofitBM,damageCostTyear.iloc[49,1],totalInitialCostOfAllComponent+retrofitBM+damageCostTyear.iloc[49,1]]
                st.session_state.MD=st.session_state.MD[~st.session_state.MD.index.duplicated(keep='last')]
                st.dataframe(st.session_state.MD.style.format('{:.2f}'))
             
      
        
    
    
# =============================================================================
# kıyaslama sayfası     
# # =============================================================================
    if rad == Navigation_Retrofit[-1]:
        st.markdown("<h2 style='text-align: center; color: black;'>Sonuçların Karşılaştırılması</h2>", unsafe_allow_html=True)
        st.markdown("""""")
        expander_bar = st.expander("Not")
        expander_bar.markdown("""
        * _Bu sayfada verilen sonuçlar bina kullanım ömrü **50 yıl** kabul edilerek raporlanmıştır._
        """)
        st.subheader("Maliyet Bileşenlerinin Kıyaslanması")

        #st.markdown(""" # h1 tag  ## h2 tag      ### h3 tag        :moon:<br>boşluk        :sunglasses:       ** bold **     _ italics _     """,True)
        
        st.markdown("""_Yapılarda sismik etkiler sonucu meydana gelen kayıpların yapı bileşenleri açısından katkısı aşağıdaki tabloda verilmiştir.
        Mevcut ve güçlendirilmiş durum için yapılarda meydana gelen kayıplara yapı bileşenlerinin katkısı ayrıca gösterilmiştir._""")

        data2 = st.session_state.buildingsComponents
        st.dataframe(data2.style.format('{:.2f}'))  
        
        dataframe2 = st.session_state.buildingsComponents.round(2)
        # Add graph data
        maliyetler = dataframe2.index
        dataframeColumn1 = dataframe2.columns[0]
        dataframeColumn2=dataframe2.columns[1]
        dataframeColumn3=dataframe2.columns[2]
        dataframeColumn4=dataframe2.columns[3]
        
        components=[dataframeColumn1, dataframeColumn2, dataframeColumn3,dataframeColumn4]
        fig2 = go.Figure(data=[
    go.Bar(name=maliyetler[0], x=components, y=[dataframe2.iloc[0,0], dataframe2.iloc[0,1], dataframe2.iloc[0,2],dataframe2.iloc[0,3]],marker_color='indianred'
),
    go.Bar(name=maliyetler[1], x=components, y=[dataframe2.iloc[1,0], dataframe2.iloc[1,1],dataframe2.iloc[1,2],dataframe2.iloc[1,3]],marker_color='lightsalmon')
])
        fig2.update_layout(legend = dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.65,
        font = dict(family = "Courier", size = 15, color = "black")),
        margin=dict(l=25, r=15, t=55, b=75),xaxis_tickfont_size=16,yaxis=dict(
        title='Kayıp (USD)',
        titlefont_size=18,
        tickfont_size=16),
        
        barmode="group",
        bargap=0.3, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1, # gap between bars of the same location coordinate.
        width=800, height=500,)
        st.write(fig2)



        
        st.subheader("Toplam Maliyetlerin Kıyaslanması")
        st.markdown("""_Yapılarda sismik etkiler, güçlendirme ve ilk yapım maliyetleri göz önüne alındığında meydana gelen maliyetlerin 
        kıyaslaması aşağıdaki tablo ve şekilde verilmiştir. Yapının mevcut ve güçlendirilmiş durum için sonuçları sunulmuştur._""")
     
        data = st.session_state.MD
        st.dataframe(data.style.format('{:.2f}'))        
        dataframe = st.session_state.MD.round(2)
        
        # Add graph data
        maliyetler = dataframe.index
        dataframeColumn1 = dataframe.columns[0]
        dataframeColumn2=dataframe.columns[1]
        dataframeColumn3=dataframe.columns[2]
        
        fig = go.Figure(go.Bar(x=maliyetler, y=dataframe[dataframeColumn1], name=dataframeColumn1, marker_color='lightsalmon',width=0.3))
        fig.add_trace(go.Bar(x=maliyetler, y=dataframe[dataframeColumn2], name=dataframeColumn2, marker_color='peachpuff',width=0.3))
        fig.add_trace(go.Bar(x=maliyetler, y=dataframe[dataframeColumn3], name=dataframeColumn3,marker_color='indianred',width=0.3))
        fig.update_layout(
        legend = dict(
        font = dict(family = "Courier", size = 15, color = "black")),
        margin=dict(l=25, r=15, t=55, b=85),xaxis_tickfont_size=16,yaxis=dict(
        title='Kayıp (USD)',
        titlefont_size=18,
        tickfont_size=16,
    ),barmode="relative",bargap=0.23, # gap between bars of adjacent location coordinates.
    width=950, height=450)
        st.write(fig)
        

        
        st.subheader("Rapor")

        min_index = dataframe[dataframe.columns[3]].idxmin()
        st.markdown("""Yapılan analizler sonucu yapının mevcut ve güçlendirilmiş durumdaki beklenen layıpları hesaplanmıştır. 
        Yapıların belirli 50 yıllık kullanım ömrü olduğu varsayıldığı durum için **_"""+min_index+"""_** daha ekonomik bulunmuştur.""")






    #                 chart_data = pd.DataFrame([st.session_state.MD['Mevcut Maliyet'][0],st.session_state.MD['Güçlendirme Maliyeti'][0],st.session_state.MD['Beklenen Maliyet'][0]],
    #      columns=["Mevcut Durum"])
     
    #                 energy_source = pd.DataFrame({
    #                     "EnergyType": ["Electricity","Gasoline","Natural Gas","Electricity","Gasoline","Natural Gas","Electricity","Gasoline","Natural Gas"],
    #                     "Price ($)":  [150,73,15,130,80,20,170,83,20],
    #                     "Date": ["2022-1-23", "2022-1-30","2022-1-5","2022-2-21", "2022-2-1","2022-2-1","2022-3-1","2022-3-1","2022-3-1"]
    #                     })
    #                 import altair as alt
    #                 bar_chart = alt.Chart(energy_source).mark_bar().encode(
    #                         y="month(Date):O",
    #                         x="sum(Price ($)):Q",
    #                         color="EnergyType:N"
    #                     )
    #                 st.altair_chart(bar_chart, use_container_width=True)              
    #                  "  
                    
                
        
    
    
if selected == "Contact":
    st.title("")
    
    st.header(":mailbox: İletişime Geçin!")


    contact_form = """
    <form action="https://formsubmit.co/mehmet10607@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="İsminiz" required>
         <input type="email" name="email" placeholder="Email Adresiniz" required>
         <textarea name="message" placeholder="Mesajınız"></textarea>
         <button type="submit">Gönder</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("style/style.css")
