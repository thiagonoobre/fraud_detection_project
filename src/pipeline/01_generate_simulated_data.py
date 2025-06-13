# Databricks notebook source
# MAGIC %md
# MAGIC # Projeto Lakehouse para Detecção de Fraudes

# COMMAND ----------

# MAGIC %md
# MAGIC O projeto consiste na criação de um pipeline de dados completo, desde a exploração dos dados até as camadas do Lakehouse, incluindo o treinamento de modelos de Machine Learning e a criação de um dashboard com os principais indicadores.
# MAGIC
# MAGIC Hoje, lidamos com muitos cenários de fraudes de cartões de crédito. O projeto irá ajudar a identificar alguns desses cenários que podem ser encontrados.
# MAGIC
# MAGIC Este projeto é baseado no livro [Reproducible Machine Learning for Credit Card detecteion Pratical handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html), que apresenta informações importantes sobre os cenários de detecção de fraude, além dos sistemas atualmente existentes e a simulação de dados para a implementação dos modelos de Machine Learning para identificação.

# COMMAND ----------

# MAGIC %md
# MAGIC Este projeto foi desenvolvido com base no [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook), de Yann-Aël Le Borgne, Wissam Siblini, Bertrand Lebichot e Gianluca Bontempi.
# MAGIC O código-fonte original está licenciado sob a GNU GPL v3.0 e os textos/imagens sob CC BY-SA 4.0.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import datetime
import time
import random


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, to_date, concat_ws
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType

sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Simulador de Dados de transação - Camada Raw

# COMMAND ----------

# MAGIC %md
# MAGIC Apresentaremos um simulador de dados de transações legítimas e fraudulentas que se aproxima da realidade. Ele reflete a profundidade do funcionamento dos dados de transação com cartão de pagamento do mundo real, tudo isso com uma configuração simples para a apresentação dos dados.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ***Features de Transação***

# COMMAND ----------

# MAGIC %md
# MAGIC objetivo será simular as características primordiais das transações. A princípio, as transações de pagamento com cartão envolvem o valor a ser pago por um cliente a um lojista em determinado momento. Os principais atributos que sumarizam uma transação são:
# MAGIC
# MAGIC 1. **ID da Transação**: Uma identificação única para a transação.
# MAGIC 2. **Data e Hora**: O momento em que a transação ocorreu.
# MAGIC 3. **ID do Cliente**: O identificador único para cada cliente.
# MAGIC 4. **ID do Terminal**: O identificador único para o comerciante (será identificado como terminal).
# MAGIC 5. **Valor**: O valor da transação.
# MAGIC 6. **Rótulo da Fraude**: Uma variável binária, com valor 0 para transações não fraudulentas ou valor 1 para transações fraudulentas.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ***Processo de geração de transações***

# COMMAND ----------

# MAGIC %md
# MAGIC A simulação **consiste** em cinco etapas principais:
# MAGIC
# MAGIC 1.  **Geração de Perfis de Clientes**: Cada cliente possui **hábitos** de consumo distintos. Isso será simulado definindo algumas propriedades para cada cliente, como seu **ID**, **geolocalização**, **frequência** de gastos e o valor gasto. As propriedades do cliente serão **representadas** em um **DataFrame** denominado `customer_profiles_table`.
# MAGIC
# MAGIC 2.  **Geração de Perfis de Terminais**: As propriedades do terminal (**lojista** ou **comerciante**) **consistirão** unicamente em sua **localização geográfica** e seu **ID**. Suas propriedades serão **representadas** em um **DataFrame** denominado `terminal_profiles_table`.
# MAGIC
# MAGIC 3.  **Associação de Perfis de Clientes com Terminais**: Um exemplo simples é associar clientes a terminais que estão dentro de um raio ***r*** de sua **localização geográfica**. Isso simplifica a premissa de que um cliente **faz transações** apenas em terminais geograficamente **próximos** à sua localização. Esta etapa **consistirá em adicionar** um recurso `list_terminals` a cada perfil de cliente, que **contém o conjunto** de terminais **que** o cliente pode usar.
# MAGIC
# MAGIC 4.  **Geração de Transações**: O simulador **irá gerar** **transações** com base nos perfis dos clientes e suas propriedades (**frequência**, valor gasto, terminais **disponíveis**). Isso resultará em um **DataFrame** de **transações**.
# MAGIC
# MAGIC 5.  **Geração de Cenários de Fraude**: Este **último** passo **trará rótulos** às **transações**, classificando-as como **legítimas ou fraudulentas**. Isso será feito seguindo três diferentes cenários de fraudes.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://raw.githubusercontent.com/thiagonoobre/fraud_detection_project/refs/heads/main/images/imagen-Processo-de-gera%C3%A7%C3%A3o-de-transa%C3%A7%C3%B5es.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1. Geração de perfis de clientes
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Todos os clientes terão os seguintes atributos:
# MAGIC
# MAGIC * **CUSTOMER\_ID**: O ID único do cliente.
# MAGIC * **x\_customer\_id** e **y\_customer\_id**: Um par de coordenadas reais em uma grade 100x100 que define a **localização geográfica** do cliente.
# MAGIC * **mean\_amount** e **std\_amount**: A média e o desvio padrão dos valores **das transações** do cliente, assumindo que os valores das transações seguem uma **distribuição normal**. O `mean_amount` será obtido a partir de uma distribuição uniforme (5, 100) e `std_amount` será definido como o `mean_amount` dividido por dois.
# MAGIC * **mean\_nb\_tx\_per\_day**: O **número** médio de **transações feitas** por dia pelo cliente, assumindo que o **número de transações** por dia segue uma **distribuição de Poisson**. Este **número** será obtido **a partir de** uma distribuição uniforme.
# MAGIC
# MAGIC A função `generate_customer_profiles_table` fornece um **Dataframe** com os perfis dos clientes. **Ela** recebe como entrada o **número** de clientes (`n_customers`) para os quais **gera** um perfil e **um** estado aleatório para reprodutibilidade.
# MAGIC
# MAGIC

# COMMAND ----------

def generate_customer_profiles_table(n_customers, random_state=0):
    
    np.random.seed(random_state)
        
    customer_id_properties=[]
    
    # Generate customer properties from random distributions 
    for customer_id in range(n_customers):
        
        x_customer_id = np.random.uniform(0,100)
        y_customer_id = np.random.uniform(0,100)
        
        mean_amount = np.random.uniform(5,100) # Arbitrary (but sensible) value 
        std_amount = mean_amount/2 # Arbitrary (but sensible) value
        
        mean_nb_tx_per_day = np.random.uniform(0,4) # Arbitrary (but sensible) value 
        
        customer_id_properties.append([customer_id,
                                      x_customer_id, y_customer_id,
                                      mean_amount, std_amount,
                                      mean_nb_tx_per_day])
        
    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                      'x_customer_id', 'y_customer_id',
                                                                      'mean_amount', 'std_amount',
                                                                      'mean_nb_tx_per_day'])
    
    return customer_profiles_table

# COMMAND ----------

n_customers = 5
customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = 0)
customer_profiles_table

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2. Geração de perfis de terminais

# COMMAND ----------

# MAGIC %md
# MAGIC Cada terminal terá **os seguintes atributos**:
# MAGIC
# MAGIC * **TERMINAL_ID**: O ID único de cada terminal.
# MAGIC * **x_terminal_id, y_terminal_id**: Par de coordenadas reais em uma grade 100x100, que define a localização geográfica do terminal.
# MAGIC
# MAGIC A função `generate_terminal_profiles_table` **é uma implementação que gera** um **Dataframe** (uma estrutura de dados tabular) com os perfis dos terminais. Ela recebe como entrada o número de terminais e, a partir desse número, gera os perfis em um estado aleatório que permite a **reprodutibilidade** (ou seja, os resultados podem ser recriados).
# MAGIC

# COMMAND ----------

def generate_terminal_profiles_table(n_terminals, random_state=0):
    
    np.random.seed(random_state)
        
    terminal_id_properties=[]
    
    # Generate terminal properties from random distributions 
    for terminal_id in range(n_terminals):
        
        x_terminal_id = np.random.uniform(0,100)
        y_terminal_id = np.random.uniform(0,100)
        
        terminal_id_properties.append([terminal_id,
                                      x_terminal_id, y_terminal_id])
                                       
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                      'x_terminal_id', 'y_terminal_id'])
    
    return terminal_profiles_table

# COMMAND ----------

n_terminals = 5
terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state = 0)
terminal_profiles_table

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3. Associação de perfis de clientes a terminais

# COMMAND ----------

# MAGIC %md
# MAGIC Neste projeto, os **clientes** só podem realizar **transações** em terminais que estejam dentro de um raio `r` de sua localização **geográfica**.
# MAGIC
# MAGIC A função `get_list_terminals_within_radius` encontra esses terminais para um perfil de cliente. Essa função receberá o perfil do cliente, um *array* (ou lista) que contém a localização geográfica de todos os terminais, e o raio `r`. Ela retornará a lista de terminais localizados dentro desse **raio** `r` para os clientes.
# MAGIC

# COMMAND ----------

def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    
    # Use numpy arrays in the following to speed up computations
    
    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id','y_customer_id']].values.astype(float)
    
    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    
    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    
    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y<r)[0])
    
    # Return the list of terminal IDs
    return available_terminals
    

# COMMAND ----------

# MAGIC %md
# MAGIC A **demonstração** abaixo é um exemplo em que usamos o perfil do **último** cliente e definimos o raio como 50.
# MAGIC
# MAGIC Nessa **demonstração**, que pode ser observada na figura, os **únicos** terminais localizados dentro da **circunferência** de raio 50 para o **último** cliente são os terminais **2 e 3**, que se encontram a um raio menor. Com base nisso, podemos **definir** em quais terminais os **clientes** podem se locomover.
# MAGIC

# COMMAND ----------

# We first get the geographical locations of all terminals as a numpy array
x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
# And get the list of terminals within radius of $50$ for the last customer
get_list_terminals_within_radius(customer_profiles_table.iloc[4], x_y_terminals=x_y_terminals, r=50)

# COMMAND ----------

# MAGIC %%capture
# MAGIC
# MAGIC terminals_available_to_customer_fig, ax = plt.subplots(figsize=(5,5))
# MAGIC
# MAGIC # Plot locations of terminals
# MAGIC ax.scatter(terminal_profiles_table.x_terminal_id.values, 
# MAGIC            terminal_profiles_table.y_terminal_id.values, 
# MAGIC            color='blue', label = 'Locations of terminals')
# MAGIC
# MAGIC # Plot location of the last customer
# MAGIC customer_id=4
# MAGIC ax.scatter(customer_profiles_table.iloc[customer_id].x_customer_id, 
# MAGIC            customer_profiles_table.iloc[customer_id].y_customer_id, 
# MAGIC            color='red',label="Location of last customer")
# MAGIC
# MAGIC ax.legend(loc = 'upper left', bbox_to_anchor=(1.05, 1))
# MAGIC
# MAGIC # Plot the region within a radius of 50 of the last customer
# MAGIC circ = plt.Circle((customer_profiles_table.iloc[customer_id].x_customer_id,
# MAGIC                    customer_profiles_table.iloc[customer_id].y_customer_id), radius=50, color='g', alpha=0.2)
# MAGIC ax.add_patch(circ)
# MAGIC
# MAGIC fontsize=15
# MAGIC
# MAGIC ax.set_title("Círculo verde: \n Terminais num raio de 50 \n do último cliente")
# MAGIC ax.set_xlim([0, 100])
# MAGIC ax.set_ylim([0, 100])
# MAGIC     
# MAGIC ax.set_xlabel('x_terminal_id', fontsize=fontsize)
# MAGIC ax.set_ylabel('y_terminal_id', fontsize=fontsize)

# COMMAND ----------

# MAGIC %md
# MAGIC **A figura ilustra:**
# MAGIC * A localização de todos os terminais (em azul).
# MAGIC * A localização do **último** cliente (em vermelho).
# MAGIC * A região dentro de um raio de 50 do **último** cliente (em verde)."

# COMMAND ----------

terminals_available_to_customer_fig

# COMMAND ----------

customer_profiles_table['available_terminals']=customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=50), axis=1)
customer_profiles_table

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4. Geração de transações

# COMMAND ----------

# MAGIC %md
# MAGIC Os perfis dos clientes contêm todas as informações fundamentais **para** gerar **transações**. **A** geração de **transações** será feita pela função `generate_transactions_table`, que recebe como entrada um perfil de cliente, uma data de início e um número de dias para os quais as transações serão geradas. Ela retornará um **DataFrame** de **transações**.
# MAGIC

# COMMAND ----------

def generate_transactions_table(customer_profile, start_date = "2018-04-01", nb_days = 10):
    
    customer_transactions = []
    
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
    
    # For all days
    for day in range(nb_days):
        
        # Random number of transactions for that day 
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        
        # If nb_tx positive, let us generate transactions
        if nb_tx>0:
            
            for tx in range(nb_tx):
                
                # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that 
                # most transactions occur during the day.
                time_tx = int(np.random.normal(86400/2, 20000))
                
                # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
                if (time_tx>0) and (time_tx<86400):
                    
                    # Amount is drawn from a normal distribution  
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    
                    # If amount negative, draw from a uniform distribution
                    if amount<0:
                        amount = np.random.uniform(0,customer_profile.mean_amount*2)
                    
                    amount=np.round(amount,decimals=2)
                    
                    if len(customer_profile.available_terminals)>0:
                        
                        terminal_id = random.choice(customer_profile.available_terminals)
                    
                        customer_transactions.append([time_tx+day*86400, day,
                                                      customer_profile.CUSTOMER_ID, 
                                                      terminal_id, amount])
            
    customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    
    if len(customer_transactions)>0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions=customer_transactions[['TX_DATETIME','CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    
    return customer_transactions  
    
    

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ampliando para um conjunto de dados maior

# COMMAND ----------

# MAGIC %md
# MAGIC Seguindo **os mesmos passos** do livro, utilizaremos a função `generate_dataset`, **a qual** realizará todos os passos anteriores.
# MAGIC
# MAGIC Essa função **receberá** como entradas:
# MAGIC * O número de clientes, terminais e dias desejados.
# MAGIC * A data de início e o raio `r`.
# MAGIC
# MAGIC Ela **devolverá** a tabela de perfis de clientes e terminais gerados, além do **DataFrame** de transações.

# COMMAND ----------

def generate_dataset(n_customers = 10000, n_terminals = 1000000, nb_days=90, start_date="2018-04-01", r=5):
    
    start_time=time.time()
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = 0)
    print("Time to generate customer profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state = 1)
    print("Time to generate terminal profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    # With Pandarallel
    #customer_profiles_table['available_terminals'] = customer_profiles_table.parallel_apply(lambda x : get_list_closest_terminals(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals']=customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    # With Pandarallel
    #transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').parallel_apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    print("Time to generate transactions: {0:.2}s".format(time.time()-start_time))
    
    # Sort transactions chronologically
    transactions_df=transactions_df.sort_values('TX_DATETIME')
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True,drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns = {'index':'TRANSACTION_ID'}, inplace = True)
    
    return (customer_profiles_table, terminal_profiles_table, transactions_df)
    

# COMMAND ----------

# MAGIC %md
# MAGIC A primeira geração terá as seguintes características:
# MAGIC
# MAGIC * 5.000 clientes
# MAGIC * 10.000 terminais
# MAGIC * 183 dias de operações (correspondentes ao **período** simulado de 2018/04/01 a 2018/09/30)
# MAGIC * Raio definido de 5 (que **corresponde** a 100 terminais **disponíveis** para cada cliente)
# MAGIC

# COMMAND ----------

(customer_profiles_table, terminal_profiles_table, transactions_df)=\
    generate_dataset(n_customers = 5000, 
                     n_terminals = 10000, 
                     nb_days=183, 
                     start_date="2018-04-01", 
                     r=5)

# COMMAND ----------

transactions_df.shape

# COMMAND ----------

transactions_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5. Geração de cenários de fraude

# COMMAND ----------

# MAGIC %md
# MAGIC Sabe, quando estamos simulando transações para **detectar fraudes**, a gente adiciona alguns "truques" ao sistema para ver se ele é esperto o suficiente para pegá-los. Pense nisso como um jogo de esconde-esconde, onde a gente esconde os "ladrões" de diferentes formas:
# MAGIC
# MAGIC **Cenário 1:** Imagina que todo roubo acima de R\$ 220 é automaticamente considerado fraude. Esse é o nosso **ladrão mais óbvio**. Ele serve pra gente ter certeza que o nosso sistema de detecção de fraudes, mesmo o mais simples, consegue pegar algo tão na cara. É tipo um teste básico: se não pegar esse, algo está muito errado!
# MAGIC
# MAGIC
# MAGIC **Cenário 2:** Aqui, a gente simula que, de repente, alguns terminais (tipo as maquininhas de cartão) são "sequestrados" por bandidos por 28 dias. Todas as transações feitas nessas maquininhas durante esse período se tornam **fraudes**. É como se um criminoso usasse uma maquininha para aplicar golpes de *phishing* (aquelas mensagens falsas pra roubar dados). Para o sistema pegar isso, ele precisa ficar de olho em quantas transações estranhas acontecem em cada maquininha. Mas, como o roubo é temporário, o sistema também precisa ser inteligente para perceber quando a ameaça para.
# MAGIC
# MAGIC **Cenário 3:** Nesse cenário, simulamos que as informações de alguns clientes são roubadas. Aí, por 14 dias, um terço das compras desses clientes têm os valores multiplicados por cinco e são marcadas como **fraude**. O cliente continua usando o cartão normalmente, mas o golpista também faz compras grandes para tirar o máximo de dinheiro. Para pegar esse tipo de fraude, o sistema precisa **conhecer o padrão de gastos de cada cliente**. Assim como no cenário anterior, ele também precisa ser esperto para entender quando o problema com o cartão passa.
# MAGIC
# MAGIC Basicamente, estamos criando diferentes tipos de fraudes artificiais para testar o quão bem o nosso sistema consegue identificá-las e se adaptar às mudanças.

# COMMAND ----------

def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    
    # By default, all transactions are genuine
    transactions_df['TX_FRAUD']=0
    transactions_df['TX_FRAUD_SCENARIO']=0
    
    # Scenario 1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD']=1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD_SCENARIO']=1
    nb_frauds_scenario_1=transactions_df.TX_FRAUD.sum()
    print("Number of frauds from scenario 1: "+str(nb_frauds_scenario_1))
    
    # Scenario 2
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+28) & 
                                                    (transactions_df.TERMINAL_ID.isin(compromised_terminals))]
                            
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD']=1
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD_SCENARIO']=2
    
    nb_frauds_scenario_2=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_1
    print("Number of frauds from scenario 2: "+str(nb_frauds_scenario_2))
    
    # Scenario 3
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+14) & 
                                                    (transactions_df.CUSTOMER_ID.isin(compromised_customers))]
        
        nb_compromised_transactions=len(compromised_transactions)
        
        
        random.seed(day)
        index_fauds = random.sample(list(compromised_transactions.index.values),k=int(nb_compromised_transactions/3))
        
        transactions_df.loc[index_fauds,'TX_AMOUNT']=transactions_df.loc[index_fauds,'TX_AMOUNT']*5
        transactions_df.loc[index_fauds,'TX_FRAUD']=1
        transactions_df.loc[index_fauds,'TX_FRAUD_SCENARIO']=3
        
                             
    nb_frauds_scenario_3=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_2-nb_frauds_scenario_1
    print("Number of frauds from scenario 3: "+str(nb_frauds_scenario_3))
    
    return transactions_df                 

# COMMAND ----------

# MAGIC %time transactions_df = add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df)

# COMMAND ----------

transactions_df.TX_FRAUD.mean()

# COMMAND ----------

transactions_df.TX_FRAUD.sum()

# COMMAND ----------

transactions_df.head()

# COMMAND ----------

def get_stats(transactions_df):
    #Number of transactions per day
    nb_tx_per_day=transactions_df.groupby(['TX_TIME_DAYS'])['CUSTOMER_ID'].count()
    #Number of fraudulent transactions per day
    nb_fraud_per_day=transactions_df.groupby(['TX_TIME_DAYS'])['TX_FRAUD'].sum()
    #Number of fraudulent cards per day
    nb_fraudcard_per_day=transactions_df[transactions_df['TX_FRAUD']>0].groupby(['TX_TIME_DAYS']).CUSTOMER_ID.nunique()
    
    return (nb_tx_per_day,nb_fraud_per_day,nb_fraudcard_per_day)

(nb_tx_per_day,nb_fraud_per_day,nb_fraudcard_per_day)=get_stats(transactions_df)

n_days=len(nb_tx_per_day)
tx_stats=pd.DataFrame({"value":pd.concat([nb_tx_per_day/50,nb_fraud_per_day,nb_fraudcard_per_day])})
tx_stats['stat_type']=["nb_tx_per_day"]*n_days+["nb_fraud_per_day"]*n_days+["nb_fraudcard_per_day"]*n_days
tx_stats=tx_stats.reset_index()

# COMMAND ----------

# MAGIC %%capture
# MAGIC
# MAGIC sns.set(style='darkgrid')
# MAGIC sns.set(font_scale=1.4)
# MAGIC
# MAGIC fraud_and_transactions_stats_fig = plt.gcf()
# MAGIC
# MAGIC fraud_and_transactions_stats_fig.set_size_inches(15, 8)
# MAGIC
# MAGIC sns_plot = sns.lineplot(x="TX_TIME_DAYS", y="value", data=tx_stats, hue="stat_type", hue_order=["nb_tx_per_day","nb_fraud_per_day","nb_fraudcard_per_day"], legend=False)
# MAGIC
# MAGIC sns_plot.set_title('Total transactions, and number of fraudulent transactions \n and number of compromised cards per day', fontsize=20)
# MAGIC sns_plot.set(xlabel = "Number of days since beginning of data generation", ylabel="Number")
# MAGIC
# MAGIC sns_plot.set_ylim([0,300])
# MAGIC
# MAGIC labels_legend = ["# transactions per day (/50)", "# fraudulent txs per day", "# fraudulent cards per day"]
# MAGIC
# MAGIC sns_plot.legend(loc='upper left', labels=labels_legend,bbox_to_anchor=(1.05, 1), fontsize=15)

# COMMAND ----------

fraud_and_transactions_stats_fig

# COMMAND ----------

# MAGIC %md
# MAGIC Essa simulação criou cerca de 10.000 transações por dia. Dentre elas, em média, 85 são fraudulentas e envolvem aproximadamente 80 cartões diferentes por dia. Um detalhe importante é que, no primeiro mês, o número de fraudes é menor. Isso acontece porque dois dos cenários simulados (os cenários 2 e 3) levam um tempo para começar a gerar fraudes — 28 e 14 dias, respectivamente.
# MAGIC
# MAGIC O conjunto de dados final ficou bem interessante: ele apresenta um forte desequilíbrio entre classes (menos de 1% das transações são fraudes), mistura variáveis numéricas e categóricas, e traz relações complexas entre os dados, com fraudes que mudam ao longo do tempo — ou seja, mais próximas do que acontece no mundo real.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6. Salvamento do conjunto de dados

# COMMAND ----------

# MAGIC %md
# MAGIC Concluídos todos os **testes** necessários para validar nossas funções e obter um entendimento claro dos DataFrames envolvidos, prosseguiremos com a geração de arquivos **CSV** a partir desses DataFrames. Esses CSVs serão então importados para o banco de dados `fraud_detection`. Faremos questão de que cada arquivo seja nomeado de forma apropriada e organizado por sua camada de dados específica, como `transactions_table`, `customer_profiles_table` ou `terminal_profiles_table`.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# tranformando em um DataFrame Spark
# transactions_table = spark.createDataFrame(transactions_df)
# customer_profiles_table = spark.createDataFrame(customer_profiles_table)
# terminal_profiles_table = spark.createDataFrame(terminal_profiles_table)


# COMMAND ----------

# Verificando o tipo de DataFrame
transactions_table = transactions_df
print(type(transactions_table))
print(type(customer_profiles_table))
print(type(terminal_profiles_table))

# COMMAND ----------

# 1. Criar o banco de dados (se não existir)
spark.sql("CREATE DATABASE IF NOT EXISTS fraud_detection")

# 2. Usar esse banco como padrão (opcional, para evitar escrever fraude_raw.nome_tabela toda hora)
# spark.catalog.setCurrentDatabase("fraud_detection")

# COMMAND ----------

dbutils.fs.ls("/Volumes/workspace/fraud_detection/raw") 

# COMMAND ----------

# criar criar pasta transactions
dbutils.fs.mkdirs("/Volumes/workspace/fraud_detection/raw/transactions")

# criar criar pasta customer
dbutils.fs.mkdirs("/Volumes/workspace/fraud_detection/raw/customer")

# criar criar pasta terminal
dbutils.fs.mkdirs("/Volumes/workspace/fraud_detection/raw/terminal")

# COMMAND ----------

transactions_table.to_csv("/Volumes/workspace/fraud_detection/raw/transactions/transactions.csv", index=False)

customer_profiles_table.to_csv("/Volumes/workspace/fraud_detection/raw/customer/customer.csv", index=False)

terminal_profiles_table.to_csv("/Volumes/workspace/fraud_detection/raw/terminal/terminal.csv", index=False)
