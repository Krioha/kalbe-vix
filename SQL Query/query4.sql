select product."Product Name" as "Nama Produk" ,
count(transaction.storeid ) as "Total Amount"
from product
inner join transaction
on product.productid=transaction.productid 
group by "Nama Produk"
order by "Total Amount" desc 