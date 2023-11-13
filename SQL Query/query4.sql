select product."Product Name" as "Nama Produk" ,
sum(transaction.totalamount ) as "Total Amount"
from product
inner join transaction
on product.productid=transaction.productid 
group by "Nama Produk"
order by "Total Amount" desc 